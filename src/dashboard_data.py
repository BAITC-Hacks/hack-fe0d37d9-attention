"""Read-only presentation adapters. No inference, weather requests, or LLM calls.

Only explicitly allowed fields cross into the UI. In particular, raw audits,
exception messages, request payloads, and provider responses never cross this
boundary. Forecast validation and deterministic summaries use existing tools.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import daily_origins
from .forecasting import ForecastContext, summarize_forecast_metrics, validate_power_forecast
from .leakage import require_utc
from .weather import PROVIDER

DISPLAY_TZ = "Asia/Almaty"
FORECAST_COMMAND = "python -m src.cli february-forecast"
METRICS_COMMAND = "python -m src.cli phase2b-backtest"
MODEL_LABELS = {
    "A_direct_curve": "A. Direct Empirical Curve",
    "B_calibrated_curve": "B. Calibrated Wind → Curve",
    "C_hist_gradient": "C. HistGradientBoosting",
    "D_hybrid_catboost": "D. Hybrid CatBoost",
}
TOOL_LABELS = {
    "create_forecast_context": "Create forecast context",
    "fetch_archived_weather": "Select eligible ECMWF run & retrieve archived weather",
    "validate_weather": "Validate weather",
    "prepare_forecast_features": "Prepare model features",
    "run_power_forecast": "Run power forecast",
    "validate_power_forecast": "Validate forecast",
    "summarize_forecast_metrics": "Calculate deterministic insights",
    "save_forecast": "Save forecast",
    "analyze_forecast_with_llm": "OpenAI analysis",
}
TIME_COLUMNS = ("forecast_origin_utc", "valid_time_utc", "weather_run_init_utc", "weather_available_at_utc")
FORECAST_COLUMNS = (
    "turbine_id", *TIME_COLUMNS, "lead_time_hours", "availability_lag_hours",
    "run_age_at_origin_hours", "model_lead_time_hours", "predicted_power",
    "wind_speed_100m_ms", "wind_speed_80m_ms", "wind_speed_120m_ms", "temperature_2m_c",
    "model_name", "model_version", "run_id", "weather_provider",
)
METRIC_COLUMNS = ("population", "month", "turbine_id", "lead_group", "model_name",
                  "N", "MAE", "RMSE", "R2", "normalized_capacity_error_pp", "bias")
STATUSES = {"passed", "failed", "running", "completed", "skipped", "incomplete", "queued", "cancelled"}


@dataclass
class Loaded:
    data: Any = None
    message: str = ""
    command: str = ""


def replay_origins() -> list[str]:
    return [stamp.isoformat() for stamp in daily_origins("2026-01-31", "2026-02-28")]


def local_time(value: Any, short: bool = False) -> str:
    if value is None:
        return "Not recorded"
    stamp = pd.Timestamp(value).tz_convert(DISPLAY_TZ)
    return stamp.strftime("%b %d, %H:%M" if short else "%Y-%m-%d %H:%M")


def _label(value: Any) -> str | None:
    if (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
            and not value.lower().startswith(("sk-", "sess-", "bearer", "openai_api_key"))):
        return value
    return None


def _run_id(value: Any) -> str | None:
    label = _label(value)
    return label if label and re.fullmatch(r"[A-Za-z0-9_-]+", label) else None


def _timestamp(value: Any) -> str | None:
    try:
        stamp = pd.Timestamp(value)
        return stamp.tz_convert("UTC").isoformat() if stamp.tzinfo and not pd.isna(stamp) else None
    except (TypeError, ValueError):
        return None


def _number(value: Any, nonnegative: bool = False) -> float | None:
    if type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0):
        return value
    return None


def _status(value: Any) -> str:
    return value if isinstance(value, str) and value in STATUSES else "not recorded"


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected an object")
    return data


def context_for(frame: pd.DataFrame, recalculation: bool = False) -> ForecastContext:
    first = frame.iloc[0]
    origin = first.forecast_origin_utc
    return ForecastContext(str(first.run_id), origin, origin.tz_convert(DISPLAY_TZ),
                           tuple(sorted(frame.turbine_id.unique())), 48, str(first.model_version), recalculation)


def load_forecast(path: Path) -> Loaded:
    if not path.is_file():
        return Loaded(message="February forecast artifacts are missing.", command=FORECAST_COMMAND)
    try:
        # Round-trip parsing preserves the exact stored IEEE-754 values; no clipping,
        # smoothing, interpolation, or model calls are permitted here.
        raw = pd.read_csv(path, float_precision="round_trip")
        if any(c.startswith(("target_", "actual_")) or c == "y_true" for c in raw):
            raise ValueError("Observed targets in an operational artifact")
        required = {"turbine_id", *TIME_COLUMNS, "lead_time_hours", "availability_lag_hours",
                    "predicted_power", "model_name", "model_version", "run_id"}
        if raw.empty or not required.issubset(raw):
            raise ValueError("Missing forecast fields")
        frame = raw.loc[:, [c for c in FORECAST_COLUMNS if c in raw]].copy()
        for name in TIME_COLUMNS:
            frame[name] = require_utc(frame[name], name)
        if not set(frame.turbine_id).issubset({"T1", "T2"}):
            raise ValueError("Unknown turbine")
        if not frame.forecast_origin_utc.isin(pd.to_datetime(replay_origins())).all():
            raise ValueError("Not a February replay origin")
        for name in ("model_name", "model_version", "run_id"):
            if not frame[name].map(_run_id if name == "run_id" else _label).notna().all():
                raise ValueError("Invalid identity")
        if "weather_provider" in frame and not frame.weather_provider.eq(PROVIDER).all():
            raise ValueError("Unexpected weather provider")
        if frame.groupby("run_id").forecast_origin_utc.nunique().gt(1).any():
            raise ValueError("Run ID reused across origins")
        for name in ("wind_speed_100m_ms", "wind_speed_80m_ms", "wind_speed_120m_ms", "temperature_2m_c"):
            if name in frame:
                if not pd.api.types.is_numeric_dtype(frame[name]) or not np.isfinite(frame[name]).all():
                    raise ValueError("Invalid saved weather values")
                if name.startswith("wind_speed") and frame[name].lt(0).any():
                    raise ValueError("Negative saved wind speed")
        for _, block in frame.groupby(["forecast_origin_utc", "run_id"], sort=False):
            if block.model_version.nunique() != 1 or block.model_name.nunique() != 1:
                raise ValueError("Mixed model identity")
            validate_power_forecast(context_for(block), block)
            for _, turbine in block.groupby("turbine_id"):
                if any(turbine[c].nunique() != 1 for c in ("weather_run_init_utc", "weather_available_at_utc", "availability_lag_hours")):
                    raise ValueError("Mixed archived runs")
        frame["forecast_origin_local"] = frame.forecast_origin_utc.dt.tz_convert(DISPLAY_TZ)
        frame["valid_time_local"] = frame.valid_time_utc.dt.tz_convert(DISPLAY_TZ)
        return Loaded(frame.sort_values(["forecast_origin_utc", "run_id", "turbine_id", "lead_time_hours"]).reset_index(drop=True))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return Loaded(message="Forecast artifact is unreadable or failed the existing forecast/provenance checks. Restore or regenerate it; values have not been repaired.", command=FORECAST_COMMAND)


def _analysis(raw: dict, metadata: dict | None = None) -> dict:
    meta = metadata if isinstance(metadata, dict) else raw.get("openai", {})
    meta = meta if isinstance(meta, dict) else {}
    usage = meta.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    ids = raw.get("selected_fact_ids", [])
    allowed_ids = {"comparison", "recalculation", "limitations"} | {
        f"{t}_{kind}" for t in ("T1", "T2") for kind in ("means", "peak", "low", "ramp_up", "ramp_down")}
    return dict(status=_status(raw.get("status", meta.get("status"))),
                model=_label(meta.get("model")) or _label(meta.get("requested_model")) or _label(raw.get("model")),
                latency_ms=_number(meta.get("latency_ms"), True),
                usage={name: value if type(value := usage.get(name)) is int and value >= 0 else None
                       for name in ("input_tokens", "output_tokens", "total_tokens")},
                called=meta.get("called") if type(meta.get("called")) is bool else None,
                selected_fact_ids=[key for key in ids if isinstance(key, str) and key in allowed_ids] if isinstance(ids, list) else [])


def load_audits(root: Path) -> Loaded:
    audits, invalid = [], 0
    for path in sorted((root / "agent_runs").glob("*.json")):
        try:
            raw = _json(path)
            run_id, origin = _run_id(raw.get("run_id")), _timestamp(raw.get("forecast_origin_utc"))
            if not run_id or origin not in replay_origins() or path.stem != run_id:
                raise ValueError("Invalid audit identity")
            turbines = raw.get("turbines")
            if not isinstance(turbines, list) or not turbines or not set(turbines).issubset({"T1", "T2"}):
                raise ValueError("Invalid turbine identity")
            steps = []
            for step in raw.get("tool_calls", []):
                if not isinstance(step, dict):
                    continue
                start, end = _timestamp(step.get("started_at")), _timestamp(step.get("ended_at"))
                duration = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() if start and end else None
                steps.append(dict(tool=step.get("tool") if step.get("tool") in TOOL_LABELS else "unrecognized_tool",
                                  status=_status(step.get("status")), duration_s=duration,
                                  error_type=_label(step.get("error_type"))))
            runs = {}
            for turbine, details in raw.get("selected_weather_runs", {}).items():
                if turbine in turbines and isinstance(details, dict):
                    runs[turbine] = dict(run=_timestamp(details.get("run")), available_at=_timestamp(details.get("available_at")),
                                         cache_hit=details.get("cache_hit") if type(details.get("cache_hit")) is bool else None,
                                         fallback_steps=_number(details.get("fallback_steps"), True))
            analysis = raw.get("analysis", {})
            audits.append(dict(run_id=run_id, forecast_origin_utc=origin, turbines=sorted(turbines),
                model_version=_label(raw.get("model_version")), status=_status(raw.get("status")),
                agent_end_time=_timestamp(raw.get("agent_end_time")), tool_calls=steps,
                selected_weather_runs=runs, analysis=_analysis(analysis if isinstance(analysis, dict) else {}, raw.get("openai")),
                warning_count=len(raw.get("warnings", [])), error_count=len(raw.get("errors", [])),
                recalculation=raw.get("recalculation") is True, previous_run_id=_run_id(raw.get("previous_run_id")),
                weather_input_changed=raw.get("weather_input_changed") if type(raw.get("weather_input_changed")) is bool else None))
        except (OSError, ValueError, TypeError, AttributeError):
            invalid += 1
    return Loaded(audits, f"{invalid} unreadable or unmatchable audit file(s) omitted; no status inferred." if invalid else "")


def matching_audit(audits: list[dict], frame: pd.DataFrame) -> dict | None:
    if frame.empty or frame.run_id.nunique() != 1 or frame.forecast_origin_utc.nunique() != 1:
        return None
    context = context_for(frame)
    for audit in audits:
        if (audit["run_id"] == context.run_id and audit["forecast_origin_utc"] == context.forecast_origin_utc.isoformat()
                and audit["turbines"] == list(context.turbines) and audit["model_version"] == context.model_version):
            # Failed attempts can still have a validated, persisted forecast. Preserve
            # their failure status, but never attach conflicting weather provenance.
            for turbine, weather in audit["selected_weather_runs"].items():
                row = frame.loc[frame.turbine_id.eq(turbine)].iloc[0]
                if weather["run"] != row.weather_run_init_utc.isoformat() or weather["available_at"] != row.weather_available_at_utc.isoformat():
                    return None
            return audit
    return None


def run_options(root: Path, origin: str, canonical: pd.DataFrame | None, audits: list[dict]) -> list[dict]:
    options = {}
    if canonical is not None:
        for run_id in canonical.loc[canonical.forecast_origin_utc.eq(pd.Timestamp(origin)), "run_id"].unique():
            options[run_id] = dict(run_id=run_id, status="saved forecast", analysis="not recorded", available=True, saved_at="")
    for audit in audits:
        if audit["forecast_origin_utc"] == origin:
            run_id = audit["run_id"]
            available = run_id in options or (root / "predictions/runs" / run_id / "forecast.csv").is_file()
            options[run_id] = dict(run_id=run_id, status=audit["status"], analysis=audit["analysis"]["status"],
                                   available=available, saved_at=audit["agent_end_time"] or "")
    return sorted(options.values(), key=lambda r: (r["available"], r["status"] != "failed", r["analysis"] == "completed", r["saved_at"]), reverse=True)


def load_run(root: Path, origin: str, run_id: str, canonical: pd.DataFrame | None) -> Loaded:
    if not _run_id(run_id) or origin not in replay_origins():
        return Loaded(message="Invalid replay run selection.")
    path = root / "predictions/runs" / run_id / "forecast.csv"
    if path.is_file():
        result = load_forecast(path)
        if result.data is None:
            return result
        frame = result.data
        if not frame.run_id.eq(run_id).all() or not frame.forecast_origin_utc.eq(pd.Timestamp(origin)).all():
            return Loaded(message="Saved forecast does not match the selected origin and run ID.")
        return result
    if canonical is not None:
        frame = canonical.loc[canonical.run_id.eq(run_id) & canonical.forecast_origin_utc.eq(pd.Timestamp(origin))].copy()
        if not frame.empty:
            return Loaded(frame)
    return Loaded(message="No saved forecast for this run. A failed agent attempt may have stopped before persistence.", command=FORECAST_COMMAND)


def forecast_summary(frame: pd.DataFrame, audit: dict | None) -> Loaded:
    try:
        # This existing pure backend tool calculates insights from persisted values;
        # the UI never recalculates predictions or reimplements their statistics.
        summary = summarize_forecast_metrics(context_for(frame, bool(audit and audit["recalculation"])), frame)
        if audit:
            for turbine, run in audit["selected_weather_runs"].items():
                if run["fallback_steps"]:
                    summary["warnings"].append(f"{turbine}: older eligible run used ({int(run['fallback_steps'])} initialization steps)")
        return Loaded(summary)
    except (ValueError, KeyError, TypeError, AttributeError):
        return Loaded(message="Deterministic insights need the complete saved weather/forecast columns.")


def load_analysis(root: Path, run_id: str, audit: dict | None, summary: dict | None) -> Loaded:
    if not _run_id(run_id) or not summary or summary["run_id"] != run_id:
        return Loaded(message="No saved AI analysis for this run.")
    try:
        analysis = audit["analysis"] if audit else _analysis(_json(root / "predictions/runs" / run_id / "analysis.json"))
        analysis = dict(analysis)
        # Existing OpenAI responses select fact IDs. Re-render these using the same
        # backend renderer; never expose arbitrary legacy text, prompts, or errors.
        from .analysis_layer import forecast_facts
        facts = forecast_facts(summary)
        ids = analysis["selected_fact_ids"]
        analysis["text"] = "\n\n".join(facts[key] for key in ids if key in facts) if analysis["status"] == "completed" else ""
        if analysis["status"] == "completed" and not analysis["text"]:
            return Loaded(analysis, "Saved analysis has no verifiable fact selection; its free-form payload is not displayed.")
        return Loaded(analysis)
    except (OSError, ValueError, KeyError, TypeError):
        return Loaded(message="No saved AI analysis for this run.")


def load_model_metrics(root: Path) -> Loaded:
    path = root / "metrics/model_comparison.csv"
    if not path.is_file():
        return Loaded(message="Saved December/January model comparison is missing.", command=METRICS_COMMAND)
    try:
        raw = pd.read_csv(path, float_precision="round_trip")
        if not set(METRIC_COLUMNS).issubset(raw):
            raise ValueError("Incomplete metric columns")
        frame = raw.loc[:, list(METRIC_COLUMNS)].copy()
        domains = dict(population={"all_complete_observed", "available_only", "full_48h_folds"},
                       month={"2025-12", "2026-01", "combined"}, turbine_id={"T1", "T2", "ALL"},
                       lead_group={"1-24h", "25-48h", "1-48h"}, model_name=set(MODEL_LABELS))
        for key, values in domains.items():
            if not frame[key].isin(values).all():
                raise ValueError("Unknown metric labels")
        for key in ("N", "MAE", "RMSE", "R2", "normalized_capacity_error_pp", "bias"):
            frame[key] = pd.to_numeric(frame[key], errors="raise")
            if not np.isfinite(frame[key].dropna()).all():
                raise ValueError("Nonfinite metric")
            if key != "R2" and frame[key].isna().any():
                raise ValueError("Missing required metric")
        if frame.N.lt(1).any() or frame.N.mod(1).ne(0).any() or frame[["MAE", "RMSE"]].lt(0).any().any():
            raise ValueError("Invalid metric counts or errors")
        if frame.empty or frame.duplicated(list(domains)).any():
            raise ValueError("Empty or duplicate metrics")
        selected = None
        selection_path = root / "models/model_selection.json"
        if selection_path.is_file():
            selection = _json(selection_path).get("selected_strategy")
            selected = selection if selection in MODEL_LABELS else None
        return Loaded(dict(metrics=frame, selected=selected))
    except (OSError, ValueError, KeyError, TypeError):
        return Loaded(message="Saved model metrics are unreadable or have an unsupported schema.", command=METRICS_COMMAND)


def artifact_fingerprint(root: Path) -> tuple:
    """Cache invalidation without opening payloads or including environment/secrets."""
    paths = [root / "predictions/february_rolling_forecast.csv", root / "metrics/model_comparison.csv",
             root / "models/model_selection.json"]
    paths += list((root / "agent_runs").glob("*.json"))
    paths += list((root / "predictions/runs").glob("*/forecast.csv"))
    paths += list((root / "predictions/runs").glob("*/analysis.json"))
    result = []
    for path in sorted(paths):
        try:
            info = path.stat()
            result.append((str(path.relative_to(root)), info.st_mtime_ns, info.st_size))
        except OSError:
            continue
    return tuple(result)
