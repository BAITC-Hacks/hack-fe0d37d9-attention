"""Features that are available at forecast issue time."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEATHER_FEATURES = [
    "wind_speed_100m",
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "wind_direction_100m",
    "surface_pressure",
]
CALENDAR_FEATURES = [
    "hour",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "lead_hours",
]
MODEL_FEATURES = WEATHER_FEATURES + CALENDAR_FEATURES + ["turbine_id"]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar and forecast-lead features from UTC timestamps."""
    result = frame.copy()
    timestamps = pd.to_datetime(result["target_timestamp"], utc=True)
    origin = pd.to_datetime(result["forecast_origin"], utc=True)
    result["hour"] = timestamps.dt.hour
    result["month"] = timestamps.dt.month
    result["day_of_year"] = timestamps.dt.dayofyear
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_of_year_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    result["lead_hours"] = ((timestamps - origin).dt.total_seconds() / 3600).astype(int)
    return result
