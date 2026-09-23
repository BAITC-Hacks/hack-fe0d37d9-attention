"""Features that are available at forecast issue time."""

from __future__ import annotations

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


WEATHER_FEATURES = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "wind_direction_100m",
    "surface_pressure",
    "u100_ms",
    "v100_ms",
]
DERIVED_WEATHER_FEATURES = [
    "wind_shear_120m_80m",
    "wind_ratio_100m_80m",
    "wind_ratio_120m_100m",
    "wind_direction_sin",
    "wind_direction_cos",
]
CALENDAR_FEATURES = [
    "hour",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "lead_time_hours",
    "model_lead_time_hours",
    "run_age_at_origin_hours",
    "weather_run_init_hour_utc",
    "lead_time_bucket",
]
MODEL_FEATURES = WEATHER_FEATURES + DERIVED_WEATHER_FEATURES + CALENDAR_FEATURES + ["turbine_id"]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add prediction-time-safe features; calendar values use Asia/Almaty civil time."""
    result = frame.copy()
    timestamps_utc = pd.to_datetime(result["valid_time_utc"], utc=True)
    origin_utc = pd.to_datetime(result["forecast_origin_utc"], utc=True)
    run_utc = pd.to_datetime(result["weather_run_init_utc"], utc=True)
    timestamps_local = timestamps_utc.dt.tz_convert(ZoneInfo("Asia/Almaty"))
    result["hour"] = timestamps_local.dt.hour
    result["month"] = timestamps_local.dt.month
    result["day_of_year"] = timestamps_local.dt.dayofyear
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_of_year_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    result["lead_time_hours"] = ((timestamps_utc - origin_utc).dt.total_seconds() / 3600).astype(int)
    result["model_lead_time_hours"] = ((timestamps_utc - run_utc).dt.total_seconds() / 3600).astype(int)
    result["run_age_at_origin_hours"] = ((origin_utc - run_utc).dt.total_seconds() / 3600).astype(int)
    result["weather_run_init_hour_utc"] = run_utc.dt.hour
    result["lead_time_bucket"] = pd.cut(
        result["lead_time_hours"], bins=[0, 6, 12, 24, 36, 48], labels=False, include_lowest=True
    ).astype(int)
    result["wind_shear_120m_80m"] = result["wind_speed_120m"] - result["wind_speed_80m"]
    result["wind_ratio_100m_80m"] = result["wind_speed_100m"] / result[
        "wind_speed_80m"
    ].clip(lower=0.1)
    result["wind_ratio_120m_100m"] = result["wind_speed_120m"] / result[
        "wind_speed_100m"
    ].clip(lower=0.1)
    direction_radians = np.deg2rad(result["wind_direction_100m"])
    result["wind_direction_sin"] = np.sin(direction_radians)
    result["wind_direction_cos"] = np.cos(direction_radians)
    # Meteorological convention: direction is where wind comes from.
    result["u100_ms"] = -result["wind_speed_100m"] * np.sin(direction_radians)
    result["v100_ms"] = -result["wind_speed_100m"] * np.cos(direction_radians)
    return result
