"""Pre-February archived-weather dataset construction and temporal diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import ARTIFACTS_DIR, DEFAULT_AVAILABILITY_LAG_HOURS, TURBINES
from .features import add_calendar_features
from .leakage import filter_outer_fold_training_rows, validate_weather_samples
from .models import (
    chronological_oof_cross_domain_power_curve_predictions,
    chronological_oof_wind_calibration,
)
from .scada import aggregate_hourly, load_all_scada
from .weather import MODEL, PROVIDER, OpenMeteoSingleRunsClient, ensure_utc


TIMEZONE_CANDIDATES: dict[str, dict[str, float | str | None]] = {
    "civil_time": {"timestamp_mode": "civil_time", "fixed_utc_offset_hours": None},
    "fixed_offset_+05:00": {
        "timestamp_mode": "fixed_offset",
        "fixed_utc_offset_hours": 5.0,
    },
    "fixed_offset_+06:00": {
        "timestamp_mode": "fixed_offset",
        "fixed_utc_offset_hours": 6.0,
    },
}

TRAINING_COLUMNS = [
    "turbine_id",
    "forecast_origin_utc",
    "weather_run_init_utc",
    "weather_available_at_utc",
    "valid_time_utc",
    "lead_time_hours",
    "model_lead_time_hours",
    "run_age_at_origin_hours",
    "wind_speed_10m_ms",
    "wind_speed_80m_ms",
    "wind_speed_100m_ms",
    "wind_speed_120m_ms",
    "wind_direction_100m_deg",
    "u100_ms",
    "v100_ms",
    "temperature_2m_c",
    "surface_pressure_hpa",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "wind_shear_120m_80m_ms",
    "wind_ratio_100m_10m",
    "target_power",
    "target_scada_wind_speed",
    "target_scada_temperature",
    "is_complete_hour",
    "suspected_unavailability",
    "has_scada_hourly_timestamp",
    "availability_lag_hours",
    "weather_model",
    "weather_provider",
]


@dataclass(frozen=True)
class WeatherArchiveBuild:
    frame: pd.DataFrame
    cache_hits: int
    cache_misses: int
    forecast_origins: int


def build_archived_weather_rows(
    origins: Iterable[pd.Timestamp | str],
    client: OpenMeteoSingleRunsClient,
    horizon_hours: int = 48,
    progress: Callable[[str], None] | None = None,
) -> WeatherArchiveBuild:
    """Retrieve each origin/turbine archive forecast once, with cache accounting."""
    unique_origins = sorted({ensure_utc(origin) for origin in origins})
    if not unique_origins:
        raise ValueError("At least one forecast origin is required")
    frames: list[pd.DataFrame] = []
    cache_hits = 0
    cache_misses = 0
    for index, origin in enumerate(unique_origins, start=1):
        for turbine in TURBINES.values():
            forecast = client.get_forecast(
                turbine.latitude, turbine.longitude, origin, horizon_hours
            )
            frame = forecast.data.copy()
            frame["turbine_id"] = turbine.turbine_id
            frame["weather_provider"] = PROVIDER
            frames.append(frame)
            cache_hits += int(forecast.cache_hit)
            cache_misses += int(not forecast.cache_hit)
        if progress is not None and (index == 1 or index % 10 == 0 or index == len(unique_origins)):
            progress(
                f"weather origins {index}/{len(unique_origins)}; "
                f"cache hits={cache_hits}, misses={cache_misses}"
            )
    result = pd.concat(frames, ignore_index=True)
    validate_weather_samples(result)
    duplicate_keys = result.duplicated(
        ["turbine_id", "forecast_origin_utc", "valid_time_utc"]
    )
    if duplicate_keys.any():
        raise ValueError("Archived weather contains duplicate turbine/origin/valid-time keys")
    expected_rows = len(unique_origins) * len(TURBINES) * horizon_hours
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} archived rows, received {len(result)}")
    return WeatherArchiveBuild(
        frame=result.sort_values(
            ["turbine_id", "forecast_origin_utc", "valid_time_utc"]
        ).reset_index(drop=True),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        forecast_origins=len(unique_origins),
    )


def hourly_scada_for_candidate(candidate_name: str) -> pd.DataFrame:
    """Load a named, explicit timestamp interpretation without changing config."""
    try:
        candidate = TIMEZONE_CANDIDATES[candidate_name]
    except KeyError as error:
        raise ValueError(f"Unknown timezone candidate: {candidate_name}") from error
    raw = load_all_scada(
        timestamp_mode=str(candidate["timestamp_mode"]),
        fixed_utc_offset_hours=candidate["fixed_utc_offset_hours"],  # type: ignore[arg-type]
    )
    return aggregate_hourly(raw)


def join_archived_weather_to_scada(
    weather_rows: pd.DataFrame, hourly_scada: pd.DataFrame
) -> pd.DataFrame:
    """Perform an exact UTC join; no nearest timestamp matching is permitted."""
    targets = hourly_scada.loc[
        :,
        [
            "timestamp",
            "turbine_id",
            "wind_speed",
            "temperature",
            "power",
            "is_complete_hour",
            "suspected_unavailability",
        ],
    ].rename(
        columns={
            "timestamp": "valid_time_utc",
            "wind_speed": "target_scada_wind_speed",
            "temperature": "target_scada_temperature",
            "power": "target_power",
        }
    )
    targets["has_scada_hourly_timestamp"] = True
    joined = weather_rows.merge(
        targets,
        on=["turbine_id", "valid_time_utc"],
        how="left",
        validate="many_to_one",
    )
    joined["has_scada_hourly_timestamp"] = joined["has_scada_hourly_timestamp"].fillna(False)
    joined["is_complete_hour"] = joined["is_complete_hour"].eq(True)
    joined["suspected_unavailability"] = joined["suspected_unavailability"].eq(True)
    return joined


def to_training_schema(joined: pd.DataFrame) -> pd.DataFrame:
    """Return the stable long-format archived-forecast training schema."""
    featured = add_calendar_features(joined)
    featured["wind_ratio_100m_10m"] = featured["wind_speed_100m"] / featured[
        "wind_speed_10m"
    ].clip(lower=0.1)
    result = featured.rename(
        columns={
            "wind_speed_10m": "wind_speed_10m_ms",
            "wind_speed_80m": "wind_speed_80m_ms",
            "wind_speed_100m": "wind_speed_100m_ms",
            "wind_speed_120m": "wind_speed_120m_ms",
            "wind_direction_100m": "wind_direction_100m_deg",
            "temperature_2m": "temperature_2m_c",
            "surface_pressure": "surface_pressure_hpa",
            "wind_shear_120m_80m": "wind_shear_120m_80m_ms",
            "day_of_year_sin": "dayofyear_sin",
            "day_of_year_cos": "dayofyear_cos",
        }
    )
    missing = set(TRAINING_COLUMNS).difference(result.columns)
    if missing:
        raise ValueError(f"Training schema is missing columns: {sorted(missing)}")
    return result.loc[:, TRAINING_COLUMNS].sort_values(
        ["turbine_id", "forecast_origin_utc", "valid_time_utc"]
    ).reset_index(drop=True)


def get_training_rows_before(
    forecast_training: pd.DataFrame, outer_forecast_origin_utc: pd.Timestamp | str
) -> pd.DataFrame:
    """Return only complete, usable labels legally available before outer origin O."""
    legal = filter_outer_fold_training_rows(forecast_training, outer_forecast_origin_utc)
    target_power = pd.to_numeric(legal["target_power"], errors="coerce")
    eligible = (
        legal["is_complete_hour"].eq(True)
        & np.isfinite(target_power)
        & ~legal["suspected_unavailability"].eq(True)
    )
    return legal.loc[eligible].copy()


def _error_metrics(
    frame: pd.DataFrame, forecast_column: str, observed_column: str
) -> dict[str, float | int]:
    valid = frame[[forecast_column, observed_column]].dropna()
    if valid.empty:
        return {
            "n": 0,
            "pearson_corr": np.nan,
            "spearman_corr": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_bias": np.nan,
            "median_bias": np.nan,
        }
    forecast = valid[forecast_column].astype(float)
    observed = valid[observed_column].astype(float)
    bias = forecast - observed
    return {
        "n": len(valid),
        "pearson_corr": float(forecast.corr(observed, method="pearson")),
        "spearman_corr": float(forecast.corr(observed, method="spearman")),
        "mae": float(mean_absolute_error(observed, forecast)),
        "rmse": float(mean_squared_error(observed, forecast) ** 0.5),
        "mean_bias": float(bias.mean()),
        "median_bias": float(bias.median()),
    }


def _scopes_and_leads(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    lead_groups = {
        "1-24h": frame["lead_time_hours"].between(1, 24),
        "25-48h": frame["lead_time_hours"].between(25, 48),
        "overall": frame["lead_time_hours"].between(1, 48),
    }
    for turbine_id in ["T1", "T2", "ALL"]:
        scoped = frame if turbine_id == "ALL" else frame.loc[frame["turbine_id"].eq(turbine_id)]
        for lead_group, mask in lead_groups.items():
            yield turbine_id, lead_group, scoped.loc[mask.loc[scoped.index]]


def timezone_alignment_table(candidate_to_joined: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare exact archived-forecast/SCADA matches for each timestamp policy."""
    rows: list[dict[str, object]] = []
    for candidate_name, table in candidate_to_joined.items():
        complete = table.loc[
            table["is_complete_hour"].eq(True)
            & table["target_scada_wind_speed"].notna()
            & table["target_scada_temperature"].notna()
        ].copy()
        for turbine, lead_group, group in _scopes_and_leads(complete):
            wind = _error_metrics(group, "wind_speed_100m", "target_scada_wind_speed")
            temperature = _error_metrics(group, "temperature_2m", "target_scada_temperature")
            rows.append(
                {
                    "timezone_mode": candidate_name,
                    "turbine": turbine,
                    "lead_group": lead_group,
                    "wind_pearson_corr": wind["pearson_corr"],
                    "wind_spearman_corr": wind["spearman_corr"],
                    "wind_mae_ms": wind["mae"],
                    "wind_rmse_ms": wind["rmse"],
                    "wind_mean_bias_ms": wind["mean_bias"],
                    "wind_median_bias_ms": wind["median_bias"],
                    "temp_pearson_corr": temperature["pearson_corr"],
                    "temp_mae_c": temperature["mae"],
                    "temp_rmse_c": temperature["rmse"],
                    "temp_mean_bias_c": temperature["mean_bias"],
                    "temp_median_bias_c": temperature["median_bias"],
                    "n": int(wind["n"]),
                }
            )
    return pd.DataFrame(rows)


def lag_correlation_table(
    weather_rows: pd.DataFrame,
    hourly_scada: pd.DataFrame,
    timezone_mode: str,
    max_abs_lag_hours: int = 8,
) -> pd.DataFrame:
    """Scan exact hourly offsets for diagnosis only; it never changes timestamps."""
    observed = hourly_scada.loc[
        :,
        ["timestamp", "turbine_id", "wind_speed", "is_complete_hour"],
    ].rename(columns={"timestamp": "scada_timestamp_utc", "wind_speed": "scada_wind_speed"})
    forecast = weather_rows.loc[
        :, ["turbine_id", "forecast_origin_utc", "valid_time_utc", "lead_time_hours", "wind_speed_100m"]
    ]
    rows: list[dict[str, object]] = []
    for lag_hours in range(-max_abs_lag_hours, max_abs_lag_hours + 1):
        shifted = forecast.copy()
        # Positive lag means the SCADA timestamp is later than forecast valid time.
        shifted["scada_timestamp_utc"] = shifted["valid_time_utc"] + pd.Timedelta(
            lag_hours, unit="h"
        )
        matched = shifted.merge(
            observed, on=["turbine_id", "scada_timestamp_utc"], how="left", validate="many_to_one"
        )
        matched = matched.loc[matched["is_complete_hour"].eq(True)]
        for turbine_id in ["T1", "T2", "ALL"]:
            group = matched if turbine_id == "ALL" else matched.loc[matched["turbine_id"].eq(turbine_id)]
            metrics = _error_metrics(group, "wind_speed_100m", "scada_wind_speed")
            rows.append(
                {
                    "timezone_mode": timezone_mode,
                    "turbine": turbine_id,
                    "lag_hours": lag_hours,
                    "shift_definition": "scada_timestamp_utc = valid_time_utc + lag_hours",
                    "wind_pearson_corr": metrics["pearson_corr"],
                    "wind_mae_ms": metrics["mae"],
                    "wind_rmse_ms": metrics["rmse"],
                    "wind_mean_bias_ms": metrics["mean_bias"],
                    "n": int(metrics["n"]),
                }
            )
    return pd.DataFrame(rows)


def weather_height_comparison(joined: pd.DataFrame, timezone_mode: str) -> pd.DataFrame:
    """Compare all requested forecast wind heights to SCADA wind after exact matching."""
    complete = joined.loc[
        joined["is_complete_hour"].eq(True) & joined["target_scada_wind_speed"].notna()
    ].copy()
    complete["wind_shear_120m_80m_ms"] = (
        complete["wind_speed_120m"] - complete["wind_speed_80m"]
    )
    complete["wind_ratio_100m_10m"] = complete["wind_speed_100m"] / complete[
        "wind_speed_10m"
    ].clip(lower=0.1)
    rows: list[dict[str, object]] = []
    for turbine_id in ["T1", "T2", "ALL"]:
        group = complete if turbine_id == "ALL" else complete.loc[complete["turbine_id"].eq(turbine_id)]
        for height in (10, 80, 100, 120):
            metrics = _error_metrics(group, f"wind_speed_{height}m", "target_scada_wind_speed")
            rows.append(
                {
                    "timezone_mode": timezone_mode,
                    "turbine": turbine_id,
                    "weather_wind_height_m": height,
                    "wind_pearson_corr": metrics["pearson_corr"],
                    "wind_spearman_corr": metrics["spearman_corr"],
                    "wind_mae_ms": metrics["mae"],
                    "wind_rmse_ms": metrics["rmse"],
                    "wind_mean_bias_ms": metrics["mean_bias"],
                    "wind_median_bias_ms": metrics["median_bias"],
                    "mean_wind_shear_120m_80m_ms": float(group["wind_shear_120m_80m_ms"].mean()),
                    "mean_wind_ratio_100m_10m": float(group["wind_ratio_100m_10m"].mean()),
                    "n": int(metrics["n"]),
                }
            )
    return pd.DataFrame(rows)


def assess_timezone_evidence(alignment: pd.DataFrame) -> dict[str, object]:
    """Apply predeclared materiality thresholds rather than picking marginal wins."""
    overall = alignment.loc[
        alignment["turbine"].eq("ALL") & alignment["lead_group"].eq("overall")
    ].sort_values(["wind_pearson_corr", "wind_mae_ms"], ascending=[False, True])
    if overall.empty:
        raise ValueError("No overall timezone-alignment rows are available")
    civil = overall.loc[overall["timezone_mode"].eq("civil_time")]
    fixed5 = overall.loc[overall["timezone_mode"].eq("fixed_offset_+05:00")]
    if not civil.empty and not fixed5.empty:
        civil_row = civil.iloc[0]
        fixed5_row = fixed5.iloc[0]
        same_recent_offset = np.isclose(
            civil_row["wind_pearson_corr"], fixed5_row["wind_pearson_corr"], atol=1e-12
        ) and np.isclose(civil_row["wind_mae_ms"], fixed5_row["wind_mae_ms"], atol=1e-12)
        if same_recent_offset:
            fixed6 = overall.loc[overall["timezone_mode"].eq("fixed_offset_+06:00")]
            fixed6_comparison: dict[str, float] = {}
            fixed6_reason = ""
            if not fixed6.empty:
                fixed6_row = fixed6.iloc[0]
                wind_corr_delta = float(
                    fixed6_row["wind_pearson_corr"] - civil_row["wind_pearson_corr"]
                )
                wind_mae_change = float(
                    (fixed6_row["wind_mae_ms"] - civil_row["wind_mae_ms"])
                    / civil_row["wind_mae_ms"]
                )
                temp_corr_delta = float(
                    fixed6_row["temp_pearson_corr"] - civil_row["temp_pearson_corr"]
                )
                temp_mae_change = float(
                    (fixed6_row["temp_mae_c"] - civil_row["temp_mae_c"])
                    / civil_row["temp_mae_c"]
                )
                fixed6_comparison = {
                    "wind_pearson_delta": wind_corr_delta,
                    "wind_mae_relative_change": wind_mae_change,
                    "temperature_pearson_delta": temp_corr_delta,
                    "temperature_mae_relative_change": temp_mae_change,
                }
                fixed6_reason = (
                    f" Fixed +06:00 changes wind correlation by {wind_corr_delta:+.4f} "
                    f"and wind MAE by {wind_mae_change:+.1%}, while temperature "
                    f"correlation changes by {temp_corr_delta:+.4f} and temperature "
                    f"MAE by {temp_mae_change:+.1%}; this mixed evidence does not "
                    "justify a production-clock change."
                )
            return {
                "selected_timezone_mode": "civil_time",
                "evidence": "inconclusive",
                "reason": (
                    "civil_time and fixed_offset_+05:00 produce identical UTC timestamps "
                    "during December 2025-January 2026; retain the documented civil default."
                    + fixed6_reason
                ),
                "fixed_offset_+06:00_comparison": fixed6_comparison,
                "thresholds": {
                    "strong_correlation_advantage": 0.08,
                    "strong_mae_improvement_fraction": 0.10,
                    "moderate_correlation_advantage": 0.04,
                    "moderate_mae_improvement_fraction": 0.05,
                },
            }
    top = overall.iloc[0]
    if len(overall) == 1:
        return {
            "selected_timezone_mode": str(top["timezone_mode"]),
            "evidence": "inconclusive",
            "reason": "Only one candidate had usable exact matches.",
        }
    runner_up = overall.iloc[1]
    corr_gain = float(top["wind_pearson_corr"] - runner_up["wind_pearson_corr"])
    mae_gain_fraction = float(
        (runner_up["wind_mae_ms"] - top["wind_mae_ms"]) / runner_up["wind_mae_ms"]
    )
    if corr_gain >= 0.08 and mae_gain_fraction >= 0.10:
        evidence = "strong"
    elif corr_gain >= 0.04 and mae_gain_fraction >= 0.05:
        evidence = "moderate"
    else:
        evidence = "inconclusive"
    return {
        "selected_timezone_mode": str(top["timezone_mode"]),
        "evidence": evidence,
        "reason": (
            f"overall wind Pearson difference={corr_gain:.4f}; "
            f"MAE improvement={mae_gain_fraction:.1%} versus next candidate."
        ),
        "thresholds": {
            "strong_correlation_advantage": 0.08,
            "strong_mae_improvement_fraction": 0.10,
            "moderate_correlation_advantage": 0.04,
            "moderate_mae_improvement_fraction": 0.05,
        },
    }


def prepare_power_curve_inputs(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create OOF direct/calibrated empirical-curve features without model selection."""
    rows = training.loc[
        training["is_complete_hour"].eq(True)
        & training["target_power"].notna()
        & training["target_scada_wind_speed"].notna()
        & ~training["suspected_unavailability"].eq(True)
    ].copy()
    rows["power"] = rows["target_power"]
    rows["calibrated_wind_estimate_oof_ms"] = chronological_oof_wind_calibration(
        rows,
        "wind_speed_100m_ms",
        "target_scada_wind_speed",
        block_size=168,
        min_history_rows=168,
    )
    rows["power_curve_direct_oof_pred"] = chronological_oof_cross_domain_power_curve_predictions(
        rows,
        "target_scada_wind_speed",
        "wind_speed_100m_ms",
        block_size=168,
        min_history_rows=168,
    )
    rows["power_curve_calibrated_oof_pred"] = chronological_oof_cross_domain_power_curve_predictions(
        rows,
        "target_scada_wind_speed",
        "calibrated_wind_estimate_oof_ms",
        block_size=168,
        min_history_rows=168,
    )
    summary: list[dict[str, object]] = []
    for turbine_id in ["T1", "T2", "ALL"]:
        group = rows if turbine_id == "ALL" else rows.loc[rows["turbine_id"].eq(turbine_id)]
        for input_name, column in {
            "direct_ecmwf_100m": "power_curve_direct_oof_pred",
            "calibrated_ecmwf_100m": "power_curve_calibrated_oof_pred",
        }.items():
            valid = group[["target_power", column]].dropna()
            summary.append(
                {
                    "turbine_id": turbine_id,
                    "power_curve_input": input_name,
                    "oof_rows": len(valid),
                    "oof_mae": float(mean_absolute_error(valid["target_power"], valid[column]))
                    if not valid.empty
                    else np.nan,
                    "oof_rmse": float(mean_squared_error(valid["target_power"], valid[column]) ** 0.5)
                    if not valid.empty
                    else np.nan,
                    "note": "OOF preparation only; not a model-selection decision.",
                }
            )
    return rows, pd.DataFrame(summary)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def write_training_dataset(training: pd.DataFrame, directory: Path) -> Path:
    """Prefer Parquet but use CSV cleanly when the optional engine is unavailable."""
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / "forecast_training.parquet"
    try:
        training.to_parquet(parquet_path, index=False)
    except (ImportError, ModuleNotFoundError):
        csv_path = directory / "forecast_training.csv"
        training.to_csv(csv_path, index=False)
        return csv_path
    return parquet_path


def build_manifest(
    training: pd.DataFrame,
    archive: WeatherArchiveBuild,
    timezone_mode: str,
    hourly_scada: pd.DataFrame,
    dataset_path: Path,
) -> dict[str, object]:
    """Create compact provenance, coverage, and cache statistics for the dataset."""
    complete = training["is_complete_hour"].eq(True)
    return {
        "dataset_path": str(dataset_path),
        "forecast_origin_range_utc": {
            "start": training["forecast_origin_utc"].min().isoformat(),
            "end": training["forecast_origin_utc"].max().isoformat(),
        },
        "valid_time_range_utc": {
            "start": training["valid_time_utc"].min().isoformat(),
            "end": training["valid_time_utc"].max().isoformat(),
        },
        "forecast_origins": archive.forecast_origins,
        "rows": len(training),
        "rows_per_turbine": {
            str(key): int(value)
            for key, value in training.groupby("turbine_id").size().items()
        },
        "lead_horizon_distribution": {
            str(key): int(value)
            for key, value in training.groupby("lead_time_hours").size().items()
        },
        "timezone_interpretation": timezone_mode,
        "weather_model": MODEL,
        "weather_provider": PROVIDER,
        "availability_lag_hours": DEFAULT_AVAILABILITY_LAG_HOURS,
        "weather_variables": [
            "wind_speed_10m_ms",
            "wind_speed_80m_ms",
            "wind_speed_100m_ms",
            "wind_speed_120m_ms",
            "wind_direction_100m_deg",
            "temperature_2m_c",
            "surface_pressure_hpa",
        ],
        "scada_coverage": {
            "complete_target_rows": int(complete.sum()),
            "unmatched_exact_timestamps": int((~training["has_scada_hourly_timestamp"]).sum()),
            "incomplete_target_rows": int((~complete).sum()),
            "suspected_unavailability_rows": int(training["suspected_unavailability"].sum()),
            "hourly_by_turbine": [
                _json_ready(row)
                for row in hourly_scada.groupby("turbine_id").agg(
                    hourly_rows=("timestamp", "size"),
                    complete_hours=("is_complete_hour", "sum"),
                    incomplete_hours=("is_complete_hour", lambda values: int((~values).sum())),
                ).reset_index().to_dict(orient="records")
            ],
        },
        "cache": {"hits": archive.cache_hits, "misses": archive.cache_misses},
    }


def phase2a_artifact_paths(root: Path = ARTIFACTS_DIR) -> dict[str, Path]:
    return {
        "alignment_csv": root / "diagnostics" / "timezone_alignment.csv",
        "alignment_json": root / "diagnostics" / "timezone_alignment.json",
        "lag_csv": root / "diagnostics" / "timezone_lag_correlation.csv",
        "height_csv": root / "diagnostics" / "weather_height_comparison.csv",
        "weather_scada_json": root / "diagnostics" / "weather_scada_alignment.json",
        "power_curve_csv": root / "diagnostics" / "power_curve_preparation.csv",
        "dataset_dir": root / "datasets",
        "manifest": root / "datasets" / "forecast_training.manifest.json",
    }
