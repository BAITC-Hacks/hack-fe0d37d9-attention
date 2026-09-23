"""Explicit temporal contracts for archived-weather forecasting."""

from __future__ import annotations

import pandas as pd


class LeakageError(ValueError):
    """Raised when a sample violates the point-in-time information boundary."""


WEATHER_TIME_COLUMNS = {
    "forecast_origin_utc",
    "weather_run_init_utc",
    "weather_available_at_utc",
    "valid_time_utc",
    "lead_time_hours",
    "availability_lag_hours",
}


def validate_weather_samples(samples: pd.DataFrame) -> None:
    """Require all archived weather inputs to have been available at issue time."""
    missing = WEATHER_TIME_COLUMNS.difference(samples.columns)
    if missing:
        raise LeakageError(f"Weather samples missing temporal columns: {sorted(missing)}")
    frame = samples.copy()
    for column in WEATHER_TIME_COLUMNS.difference(
        {"lead_time_hours", "availability_lag_hours"}
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    lead = pd.to_numeric(frame["lead_time_hours"], errors="raise")
    availability_lag = pd.to_numeric(frame["availability_lag_hours"], errors="raise")
    computed_available_at = frame["weather_run_init_utc"] + pd.to_timedelta(
        availability_lag, unit="h"
    )
    invalid = (
        frame["valid_time_utc"].le(frame["forecast_origin_utc"])
        | ~lead.between(1, 48)
        | availability_lag.lt(0)
        | frame["weather_run_init_utc"].gt(frame["forecast_origin_utc"])
        | frame["weather_available_at_utc"].gt(frame["forecast_origin_utc"])
        | computed_available_at.ne(frame["weather_available_at_utc"])
        | computed_available_at.gt(frame["forecast_origin_utc"])
    )
    if invalid.any():
        raise LeakageError(
            f"{int(invalid.sum())} weather samples violate forecast-time availability"
        )


def filter_outer_fold_training_rows(
    rows: pd.DataFrame, outer_forecast_origin_utc: pd.Timestamp | str
) -> pd.DataFrame:
    """Return only historical feature/target pairs legal at an outer origin.

    The target must have occurred before the outer decision time. The source
    forecast must also have been issued before it: otherwise its weather values
    could reflect information unavailable at the outer decision time.
    """
    required = {"forecast_origin_utc", "valid_time_utc"}
    missing = required.difference(rows.columns)
    if missing:
        raise LeakageError(f"Training rows missing temporal columns: {sorted(missing)}")
    cutoff = pd.Timestamp(outer_forecast_origin_utc)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    valid_times = pd.to_datetime(rows["valid_time_utc"], utc=True, errors="raise")
    source_origins = pd.to_datetime(rows["forecast_origin_utc"], utc=True, errors="raise")
    return rows.loc[valid_times.lt(cutoff) & source_origins.lt(cutoff)].copy()
