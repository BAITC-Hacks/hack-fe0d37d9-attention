"""Weather-feature construction and leakage-safe walk-forward evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import ARTIFACTS_DIR, PRIMARY_WIND_FEATURE, TURBINES
from .features import add_calendar_features
from .models import CatBoostPowerModel, EmpiricalPowerCurve
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
    targets = hourly_scada.loc[
        hourly_scada["is_complete_hour"], ["timestamp", "turbine_id", "power"]
    ].rename(columns={"timestamp": "target_timestamp"})
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
                on=["target_timestamp", "turbine_id"],
                how="left",
                validate="one_to_one",
            )
            frames.append(weather)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def leakage_safe_training_rows(
    weather_target_frame: pd.DataFrame, forecast_origin: pd.Timestamp | str
) -> pd.DataFrame:
    """Guarantee no target or originating forecast is at/after the evaluation time."""
    cutoff = ensure_utc(forecast_origin)
    origins = pd.to_datetime(weather_target_frame["forecast_origin"], utc=True)
    target_times = pd.to_datetime(weather_target_frame["target_timestamp"], utc=True)
    return weather_target_frame.loc[
        origins.lt(cutoff) & target_times.lt(cutoff) & weather_target_frame["power"].notna()
    ].copy()


def _metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    truth = rows["power"].to_numpy(dtype=float)
    prediction = rows["predicted_power"].to_numpy(dtype=float)
    return {
        "n": len(rows),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(rows) >= 2 else np.nan,
    }


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics by model and requested 1--24h / 25--48h / overall bands."""
    bands = {
        "1-24h": predictions["lead_hours"].between(1, 24),
        "25-48h": predictions["lead_hours"].between(25, 48),
        "overall": predictions["lead_hours"].between(1, 48),
    }
    summary: list[dict[str, object]] = []
    for (model_name, turbine_id), group in predictions.groupby(["model_name", "turbine_id"]):
        for horizon_band, mask in bands.items():
            summary.append(
                {
                    "model_name": model_name,
                    "turbine_id": turbine_id,
                    "horizon_band": horizon_band,
                    **_metrics(group.loc[mask]),
                }
            )
    return pd.DataFrame(summary)


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
        training_origins = pd.date_range(
            origin - pd.Timedelta(training_days, unit="d"),
            origin - pd.Timedelta(1, unit="d"),
            freq="1d",
            tz="UTC",
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
                    "forecast_origin",
                    "selected_weather_run",
                    "target_timestamp",
                    "lead_hours",
                    "turbine_id",
                    "power",
                    "weather_model",
                ],
            ].copy()
            result["predicted_power"] = values
            result["model_name"] = model_name
            result["selected_weather_run"] = pd.to_datetime(
                result["selected_weather_run"], utc=True
            )
            prediction_frames.append(result)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.loc[
        :,
        [
            "forecast_origin",
            "selected_weather_run",
            "target_timestamp",
            "lead_hours",
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
