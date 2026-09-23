"""Gap-aware hourly timeline helpers for any future lag or rolling features."""

from __future__ import annotations

import pandas as pd


def reindex_hourly_with_segments(hourly_scada: pd.DataFrame) -> pd.DataFrame:
    """Reindex every turbine to complete UTC hours and label contiguous valid segments."""
    frames: list[pd.DataFrame] = []
    for turbine_id, group in hourly_scada.groupby("turbine_id", sort=True):
        group = group.sort_values("timestamp").set_index("timestamp")
        expected = pd.date_range(group.index.min(), group.index.max(), freq="1h", tz="UTC")
        timeline = group.reindex(expected)
        timeline.index.name = "timestamp"
        timeline["turbine_id"] = turbine_id
        timeline["is_complete_hour"] = timeline["is_complete_hour"].eq(True)
        missing = ~timeline["is_complete_hour"]
        segment_start = timeline["is_complete_hour"] & (
            timeline["is_complete_hour"].shift(fill_value=False).eq(False)
        )
        segment_number = segment_start.cumsum()
        timeline["segment_id"] = segment_number.where(~missing).astype("Int64")
        frames.append(timeline.reset_index())
    return pd.concat(frames, ignore_index=True)


def segment_safe_shift(
    segmented_hourly: pd.DataFrame, column: str, periods: int = 1
) -> pd.Series:
    """Shift only inside contiguous complete-hour segments; missing gaps break lags."""
    if column not in segmented_hourly:
        raise KeyError(column)
    return segmented_hourly.groupby(["turbine_id", "segment_id"], dropna=True)[column].shift(periods)


def segment_safe_rolling_mean(
    segmented_hourly: pd.DataFrame, column: str, window: int
) -> pd.Series:
    """Rolling mean constrained to a segment, never carried across missing hours."""
    if window < 1:
        raise ValueError("window must be positive")
    return (
        segmented_hourly.groupby(["turbine_id", "segment_id"], dropna=True)[column]
        .transform(lambda values: values.rolling(window=window, min_periods=window).mean())
    )
