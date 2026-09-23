"""One tool-based orchestrator; all temporal decisions stay in deterministic Python."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .analysis_layer import ANALYSIS_FAILURE, OpenAIAnalysisError, analyze_forecast_with_llm, skipped_openai
from .config import ARTIFACTS_DIR
from .evaluation import daily_origins
from .forecasting import (
    ForecastResult, create_forecast_context, fetch_archived_weather, validate_weather,
    prepare_forecast_features, run_power_forecast, validate_power_forecast,
    summarize_forecast_metrics, save_forecast, load_frozen_models, persist_february_batch,
)
from .training_data import write_json
from .weather import OpenMeteoSingleRunsClient


class ForecastOrchestrator:
    """Executable state machine with failure gates, bounded retrieval decisions, and audit.

    It works without an LLM. Optional OpenAI analysis sees only a deep-copied
    summary after validation/persistence and has no numerical tool authority.
    """

    def __init__(self, client=None, manifest=None, models=None, root: Path=ARTIFACTS_DIR,
                 analysis_tool=analyze_forecast_with_llm):
        if manifest is None or models is None:
            manifest, models = load_frozen_models(root)
        self.manifest, self.models, self.root = manifest, models, Path(root)
        self.client = client or OpenMeteoSingleRunsClient(availability_lag_hours=manifest["availability_lag_hours"])
        self.analysis_tool = analysis_tool

    def _prior_success(self, origin: pd.Timestamp, turbines: tuple[str, ...]) -> dict | None:
        candidates = []
        for path in (self.root / "agent_runs").glob("*.json"):
            record = json.loads(path.read_text("utf-8"))
            if (record.get("status") == "completed" and record.get("forecast_origin_utc") == origin.isoformat()
                    and record.get("turbines") == list(turbines)):
                candidates.append(record)
        return max(candidates, key=lambda record: record["agent_end_time"]) if candidates else None

    def run(self, origin: str | pd.Timestamp, turbines=("T1", "T2"), no_llm: bool=False) -> ForecastResult:
        audit = dict(run_id=uuid4().hex, agent_start_time=pd.Timestamp.now(tz="UTC").isoformat(),
                     tool_calls=[], warnings=[], errors=[], status="running", recalculation=False,
                     model_version=self.manifest["model_version"], model_artifacts=self.manifest["files"],
                     turbines=list(turbines), openai=skipped_openai("Analysis step not reached"))
        event_start = len(self.client.events)

        def call(name, function, *args, **kwargs):
            record = dict(tool=name, started_at=pd.Timestamp.now(tz="UTC").isoformat(), status="running")
            audit["tool_calls"].append(record)
            try:
                value = function(*args, **kwargs)
                record["status"] = "passed"
                return value
            except Exception as error:
                # An arbitrary analysis adapter may raise with credentials or prompts
                # in its message; never serialize that message at this outer boundary.
                message = ANALYSIS_FAILURE if name == "analyze_forecast_with_llm" else str(error)
                record.update(status="failed", error_type=type(error).__name__, error=message)
                raise
            finally:
                record["ended_at"] = pd.Timestamp.now(tz="UTC").isoformat()

        try:
            context = call("create_forecast_context", create_forecast_context, origin, tuple(turbines), self.manifest)
            audit.update(run_id=context.run_id, forecast_origin_utc=context.forecast_origin_utc.isoformat(),
                         forecast_origin_local=context.forecast_origin_local.isoformat())
            prior = self._prior_success(context.forecast_origin_utc, tuple(turbines))
            if prior:
                from dataclasses import replace
                context = replace(context, recalculation=True)
                audit.update(recalculation=True, previous_run_id=prior["run_id"])
            if self.client.availability_lag_hours != self.manifest["availability_lag_hours"]:
                raise ValueError("Weather availability policy differs from frozen model manifest")
            print(f"Origin {context.forecast_origin_local.isoformat()} / {context.forecast_origin_utc.isoformat()}", flush=True)
            batch = call("fetch_archived_weather", fetch_archived_weather, context, self.client)
            provenance = call("validate_weather", validate_weather, context, batch)
            audit["selected_weather_runs"] = provenance["runs"]
            audit["weather_input_changed"] = bool(prior and any(
                provenance["runs"][t]["raw_sha256"] != prior["selected_weather_runs"][t]["raw_sha256"] for t in turbines))
            for turbine, details in provenance["runs"].items():
                print(f"{turbine}: ECMWF run {details['run']}; weather validation passed; cache={details['cache_hit']}", flush=True)
                if details["fallback_steps"]:
                    audit["warnings"].append(f"{turbine}: older eligible run used ({details['fallback_steps']} initialization steps)")
            features = call("prepare_forecast_features", prepare_forecast_features, context, batch)
            forecast = call("run_power_forecast", run_power_forecast, context, features, self.models)
            call("validate_power_forecast", validate_power_forecast, context, forecast)
            summary = call("summarize_forecast_metrics", summarize_forecast_metrics, context, forecast)
            summary["warnings"].extend(audit["warnings"])
            result = call("save_forecast", save_forecast, context, forecast, summary, self.root)
            write_json({t: asdict(f.metadata) for t, f in batch.forecasts.items()}, result.path.parent / "weather_provenance.json")
            audit.update(forecast_output_path=str(result.path), forecast_rows=result.rows, warnings=summary["warnings"])
            for turbine, values in summary["turbines"].items():
                print(f"{turbine}: mean normalized power 24h={values['mean_power_24h']:.4f}, 48h={values['mean_power_48h']:.4f}", flush=True)
            if no_llm or not os.getenv("OPENAI_API_KEY"):
                reason = "--no-llm" if no_llm else "OPENAI_API_KEY not configured"
                analysis = dict(status="skipped", reason=reason, openai=skipped_openai(reason))
            else:
                try:
                    analysis = call("analyze_forecast_with_llm", self.analysis_tool, copy.deepcopy(summary))
                except OpenAIAnalysisError as error:
                    analysis = dict(status="failed", reason=ANALYSIS_FAILURE, openai=error.openai)
                    audit["warnings"].append(analysis["reason"])
                except Exception:
                    analysis = dict(status="failed", reason="Optional analysis failed; validated forecast unchanged")
                    audit["warnings"].append(analysis["reason"])
            # Legacy/custom analysis adapters may not supply request instrumentation.
            # Do not claim a real API call just because such an adapter returned text.
            audit["openai"] = analysis.get("openai", dict(called=None, status=analysis["status"],
                                                        reason="Analysis adapter supplied no OpenAI request metadata"))
            write_json(analysis, result.path.parent / "analysis.json")
            audit.update(analysis=analysis, status="completed")
            print(analysis.get("text", f"AI analysis: {analysis['status']} ({analysis.get('reason', '')})"), flush=True)
            return result
        except Exception as error:
            audit.update(status="failed", errors=[dict(type=type(error).__name__, message=str(error))])
            raise
        finally:
            audit["weather_decisions"] = self.client.events[event_start:]
            audit["agent_end_time"] = pd.Timestamp.now(tz="UTC").isoformat()
            write_json(audit, self.root / "agent_runs" / f"{audit['run_id']}.json")


def replay_february() -> dict:
    """Replay the same tools with network access explicitly forbidden."""
    client = OpenMeteoSingleRunsClient(cache_only=True)
    orchestrator = ForecastOrchestrator(client=client)
    results = [orchestrator.run(origin, no_llm=True) for origin in daily_origins("2026-01-31", "2026-02-28")]
    qc = persist_february_batch(results, prefix="february_replay")
    original = pd.read_csv(ARTIFACTS_DIR / "predictions/february_rolling_forecast.csv")
    replay = pd.read_csv(ARTIFACTS_DIR / "predictions/february_replay_rolling_forecast.csv")
    keys = ["turbine_id", "forecast_origin_utc", "valid_time_utc"]
    original, replay = (frame.sort_values(keys).reset_index(drop=True) for frame in (original, replay))
    if not original[keys].equals(replay[keys]) or not np.array_equal(original.predicted_power, replay.predicted_power):
        raise ValueError("Cache-only replay differs from canonical February forecast")
    counts = pd.Series([event["action"] for event in client.events]).value_counts().to_dict()
    if counts.get("network_request", 0):
        raise AssertionError("Replay made a network request")
    qc.update(identical_to_canonical=True, cache_statistics=counts, llm_enabled=False)
    write_json(qc, ARTIFACTS_DIR / "diagnostics/february_replay_forecast_qc.json")
    print(f"Replay exactly reproduces canonical predictions; cache statistics={counts}", flush=True)
    return qc
