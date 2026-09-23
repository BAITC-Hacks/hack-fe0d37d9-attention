"""Weather-feature construction and leakage-safe walk-forward evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    ARTIFACTS_DIR,
    FORECAST_ORIGIN_CONVENTION,
    PRIMARY_WIND_FEATURE,
    SCADA_CIVIL_TIMEZONE,
    TURBINES,
)
from .features import add_calendar_features
from .leakage import filter_outer_fold_training_rows, validate_weather_samples
from .models import (
    CatBoostPowerModel,
    EmpiricalPowerCurve,
    HistGradientBoostingTargetModel,
    TwoStageWeatherToPowerModel,
    clip_predictions,
)
from .weather import OpenMeteoSingleRunsClient, ensure_utc


def build_weather_target_frame(
    hourly_scada: pd.DataFrame,
    origins: Iterable[pd.Timestamp | str],
    client: OpenMeteoSingleRunsClient,
    horizon_hours: int = 48,
) -> pd.DataFrame:
    """Join archived forecast features to complete SCADA targets when available.

    Missing or incomplete targets remain absent from this frame. Weather retrieval
    itself never uses SCADA weather observations as a substitute.
    """
    # These SCADA fields are labels/diagnostics after the forecast target time.
    # They are deliberately not included in MODEL_FEATURES or forecast-time calls.
    targets = hourly_scada.loc[
        hourly_scada["is_complete_hour"],
        [
            "timestamp",
            "turbine_id",
            "wind_speed",
            "temperature",
            "power",
            "suspected_unavailability",
        ],
    ].rename(
        columns={
            "timestamp": "valid_time_utc",
            "wind_speed": "actual_scada_wind_speed",
            "temperature": "actual_scada_temperature",
            "power": "actual_scada_power",
        }
    )
    frames: list[pd.DataFrame] = []

    for raw_origin in origins:
        origin = ensure_utc(raw_origin)
        for turbine in TURBINES.values():
            archived = client.get_forecast(
                turbine.latitude, turbine.longitude, origin, horizon_hours
            )
            weather = archived.data.copy()
            weather["turbine_id"] = turbine.turbine_id
            weather = add_calendar_features(weather)
            weather = weather.merge(
                targets.loc[targets["turbine_id"].eq(turbine.turbine_id)],
                on=["valid_time_utc", "turbine_id"],
                how="left",
                validate="one_to_one",
            )
            # Alias retained for model evaluation; actual_scada_power remains
            # explicit in persisted diagnostic/training tables.
            weather["power"] = weather["actual_scada_power"]
            frames.append(weather)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    validate_weather_samples(result)
    return result


def leakage_safe_training_rows(
    weather_target_frame: pd.DataFrame, forecast_origin: pd.Timestamp | str
) -> pd.DataFrame:
    """Return only labels and archived weather legally known at an outer origin."""
    validate_weather_samples(weather_target_frame)
    legal = filter_outer_fold_training_rows(weather_target_frame, forecast_origin)
    legal = legal.loc[legal["power"].notna()].copy()
    if "suspected_unavailability" in legal:
        legal = legal.loc[~legal["suspected_unavailability"].fillna(False)].copy()
    return legal


def _metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "rated_capacity_mae_percent": np.nan,
            "bias": np.nan,
        }
    truth = rows["power"].to_numpy(dtype=float)
    prediction = rows["predicted_power"].to_numpy(dtype=float)
    return {
        "n": len(rows),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(rows) >= 2 else np.nan,
        "rated_capacity_mae_percent": float(mean_absolute_error(truth, prediction) * 100),
        "bias": float(np.mean(prediction - truth)),
    }


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Held-out metrics by month, model, turbine, and horizon band."""
    working = predictions.copy()
    working["validation_month"] = pd.to_datetime(
        working["forecast_origin_utc"], utc=True
    ).dt.strftime("%Y-%m")
    bands = {
        "1-24h": (1, 24),
        "25-48h": (25, 48),
        "overall": (1, 48),
    }
    summary: list[dict[str, object]] = []
    for horizon_band, (start_hour, end_hour) in bands.items():
        band = working.loc[working["lead_time_hours"].between(start_hour, end_hour)]
        for (month, model_name, turbine_id), group in band.groupby(
            ["validation_month", "model_name", "turbine_id"]
        ):
            summary.append(
                {
                    "validation_month": month,
                    "model_name": model_name,
                    "turbine_id": turbine_id,
                    "horizon_band": horizon_band,
                    **_metrics(group),
                }
            )
        for (month, model_name), group in band.groupby(["validation_month", "model_name"]):
            summary.append(
                {
                    "validation_month": month,
                    "model_name": model_name,
                    "turbine_id": "ALL",
                    "horizon_band": horizon_band,
                    **_metrics(group),
                }
            )
    return pd.DataFrame(summary)


def daily_origins(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    convention: str = FORECAST_ORIGIN_CONVENTION,
) -> pd.DatetimeIndex:
    """Return daily origins under the explicit configured civil-time convention."""
    if convention != "almaty_midnight":
        raise ValueError(f"Unsupported forecast-origin convention: {convention!r}")

    def _civil_date(value: pd.Timestamp | str) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(ZoneInfo(SCADA_CIVIL_TIMEZONE))
        return stamp.normalize().tz_localize(None)

    local_origins = pd.date_range(
        _civil_date(start), _civil_date(end), freq="1d", tz=ZoneInfo(SCADA_CIVIL_TIMEZONE)
    )
    return local_origins.tz_convert("UTC")


def run_historical_walk_forward(
    hourly_scada: pd.DataFrame,
    historical_forecast_table: pd.DataFrame,
    validation_origins: Iterable[pd.Timestamp | str],
    catboost_iterations: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily expanding-window evaluation on a pre-built archived forecast table.

    Each fit sees only examples whose forecast was issued before the evaluation
    origin *and* whose power target was known before that origin. This permits a
    large cached training table without introducing future-information leakage.
    """
    complete_scada = hourly_scada.loc[hourly_scada["is_complete_hour"]].copy()
    prediction_frames: list[pd.DataFrame] = []
    for raw_origin in validation_origins:
        origin = ensure_utc(raw_origin)
        training = leakage_safe_training_rows(historical_forecast_table, origin)
        evaluation = historical_forecast_table.loc[
            pd.to_datetime(historical_forecast_table["forecast_origin_utc"], utc=True).eq(origin)
            & historical_forecast_table["power"].notna()
        ].copy()
        if training.empty or evaluation.empty:
            continue
        prior_scada = complete_scada.loc[complete_scada["timestamp"].lt(origin)]
        power_curve = EmpiricalPowerCurve().fit(prior_scada)
        catboost = CatBoostPowerModel(iterations=catboost_iterations).fit(training)
        hist_gradient = HistGradientBoostingTargetModel("power").fit(training)
        two_stage = TwoStageWeatherToPowerModel().fit(training, prior_scada)
        model_predictions = {
            "empirical_scada_power_curve": power_curve.predict(
                evaluation, PRIMARY_WIND_FEATURE
            ),
            "catboost_archived_ecmwf": catboost.predict(evaluation),
            "hist_gradient_archived_ecmwf": clip_predictions(hist_gradient.predict(evaluation)),
            "two_stage_ecmwf_to_wind_curve": two_stage.predict(evaluation),
        }
        for model_name, values in model_predictions.items():
            result = evaluation.loc[
                :,
                [
                    "forecast_origin_utc",
                    "weather_run_init_utc",
                    "weather_available_at_utc",
                    "valid_time_utc",
                    "lead_time_hours",
                    "turbine_id",
                    "power",
                    "actual_scada_wind_speed",
                    "actual_scada_temperature",
                    "wind_speed_100m",
                    "hour",
                    "weather_model",
                ],
            ].copy()
            result["predicted_power"] = clip_predictions(values)
            result["model_name"] = model_name
            result["training_row_count"] = len(training)
            prediction_frames.append(result)
    if not prediction_frames:
        raise ValueError("No valid pre-February walk-forward predictions were produced")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.loc[
        :,
        [
            "forecast_origin_utc",
            "weather_run_init_utc",
            "weather_available_at_utc",
            "valid_time_utc",
            "lead_time_hours",
            "turbine_id",
            "predicted_power",
            "power",
            "actual_scada_wind_speed",
            "actual_scada_temperature",
            "wind_speed_100m",
            "hour",
            "model_name",
            "weather_model",
            "training_row_count",
        ],
    ]
    return predictions, summarize_metrics(predictions), error_analysis(predictions)


def error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Held-out forecast errors broken down by operationally useful slices."""
    working = predictions.copy()
    working["error"] = working["predicted_power"] - working["power"]
    working["wind_speed_bucket"] = pd.cut(
        working["actual_scada_wind_speed"],
        bins=[-np.inf, 4, 8, 12, np.inf],
        labels=["<4", "4-8", "8-12", "12+"],
    )
    working["generation_bucket"] = pd.cut(
        working["power"],
        bins=[-0.001, 0.1, 0.7, 1.001],
        labels=["low_0-0.1", "medium_0.1-0.7", "high_0.7-1.0"],
    )
    working["lead_band"] = np.where(
        working["lead_time_hours"] <= 24, "1-24h", "25-48h"
    )
    slices = {
        "wind_speed_bucket": ["model_name", "turbine_id", "wind_speed_bucket"],
        "lead_horizon": ["model_name", "turbine_id", "lead_band"],
        "hour_of_day": ["model_name", "turbine_id", "hour"],
        "generation_bucket": ["model_name", "turbine_id", "generation_bucket"],
    }
    rows: list[dict[str, object]] = []
    for dimension, group_columns in slices.items():
        for keys, group in working.groupby(group_columns, observed=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            row = {"dimension": dimension, **dict(zip(group_columns, key_values, strict=True))}
            row.update(
                {
                    "n": len(group),
                    "mae": float(np.mean(np.abs(group["error"]))),
                    "rmse": float(np.sqrt(np.mean(np.square(group["error"]))),),
                    "bias": float(np.mean(group["error"])),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_walk_forward(
    hourly_scada: pd.DataFrame,
    evaluation_origins: Iterable[pd.Timestamp | str],
    client: OpenMeteoSingleRunsClient,
    training_days: int = 14,
    horizon_hours: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both baselines using only per-origin information available in time.

    Historical forecast features are retrieved for a bounded daily training window
    to keep the MVP fast. At each evaluation origin, both forecast-origin and
    target-time cutoffs are applied before CatBoost fitting.
    """
    if training_days < 2:
        raise ValueError("training_days must be at least 2")
    complete_scada = hourly_scada.loc[hourly_scada["is_complete_hour"]].copy()
    prediction_frames: list[pd.DataFrame] = []

    for raw_origin in evaluation_origins:
        origin = ensure_utc(raw_origin)
        training_origins = daily_origins(
            origin - pd.Timedelta(training_days, unit="d"),
            origin - pd.Timedelta(1, unit="d"),
        )
        train_candidates = build_weather_target_frame(
            hourly_scada, training_origins, client, horizon_hours
        )
        train_rows = leakage_safe_training_rows(train_candidates, origin)
        if train_rows.empty:
            raise ValueError(f"No leakage-safe archived-weather training rows before {origin}")
        evaluation = build_weather_target_frame(hourly_scada, [origin], client, horizon_hours)
        evaluation = evaluation.loc[evaluation["power"].notna()].copy()
        if evaluation.empty:
            raise ValueError(f"No complete SCADA targets available after {origin}")

        power_curve = EmpiricalPowerCurve().fit(
            complete_scada.loc[complete_scada["timestamp"].lt(origin)]
        )
        catboost = CatBoostPowerModel().fit(train_rows)
        model_predictions = {
            "empirical_power_curve": power_curve.predict(evaluation, PRIMARY_WIND_FEATURE),
            "catboost_archived_weather": catboost.predict(evaluation),
        }
        for model_name, values in model_predictions.items():
            result = evaluation.loc[
                :,
                [
                    "forecast_origin_utc",
                    "weather_run_init_utc",
                    "weather_available_at_utc",
                    "valid_time_utc",
                    "lead_time_hours",
                    "turbine_id",
                    "power",
                    "weather_model",
                ],
            ].copy()
            result["predicted_power"] = values
            result["model_name"] = model_name
            result["weather_run_init_utc"] = pd.to_datetime(
                result["weather_run_init_utc"], utc=True
            )
            prediction_frames.append(result)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.loc[
        :,
        [
            "forecast_origin_utc",
            "weather_run_init_utc",
            "weather_available_at_utc",
            "valid_time_utc",
            "lead_time_hours",
            "turbine_id",
            "predicted_power",
            "power",
            "model_name",
            "weather_model",
        ],
    ]
    return predictions, summarize_metrics(predictions)


def persist_predictions(predictions: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist the auditable forecast schema, without overwriting source data."""
    destination = path or ARTIFACTS_DIR / "walk_forward_predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    return destination
