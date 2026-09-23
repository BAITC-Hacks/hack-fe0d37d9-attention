"""SCADA loading and safe hourly aggregation without gap interpolation."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import (
    PROJECT_ROOT,
    SCADA_CIVIL_TIMEZONE,
    SCADA_FIXED_UTC_OFFSET_HOURS,
    SCADA_TIMESTAMP_MODE,
    TURBINES,
)

RAW_COLUMNS = {
    "Статистическое время": "timestamp",
    "Средняя скорость ветра(m/s)": "wind_speed",
    "Нормализованная активная мощность": "power",
    "Средняя температура окружающей среды(°C)": "temperature",
}
REQUIRED_COLUMNS = set(RAW_COLUMNS)


class ScadaDataError(ValueError):
    """Raised when a SCADA source file does not meet the expected contract."""


def resolve_turbine_path(turbine_id: str, root: Path = PROJECT_ROOT) -> Path:
    """Resolve exactly one original source CSV without copying or modifying it."""
    try:
        turbine = TURBINES[turbine_id]
    except KeyError as error:
        raise ScadaDataError(f"Unknown turbine_id: {turbine_id}") from error
    matches = list(root.glob(turbine.source_glob))
    if len(matches) != 1:
        raise ScadaDataError(
            f"Expected exactly one source matching {turbine.source_glob!r}; found {len(matches)}"
        )
    return matches[0]


def _as_utc(
    series: pd.Series,
    timestamp_mode: str,
    fixed_utc_offset_hours: float | None,
) -> pd.Series:
    """Interpret naive SCADA clocks using one explicit configured convention."""
    parsed = pd.to_datetime(series, errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_convert("UTC")
    if timestamp_mode == "civil_time":
        # Kazakhstan moved Almaty from UTC+06 to UTC+05 at 2024-03-01.
        # The raw source contains no fold marker for the repeated 23:00 hour.
        # It records that hour once, so we assign it to the earlier (+06) civil
        # occurrence. This narrow, documented clock-fold policy is distinct from
        # inferring the overall source timezone convention.
        return parsed.dt.tz_localize(
            ZoneInfo(SCADA_CIVIL_TIMEZONE), ambiguous=True, nonexistent="raise"
        ).dt.tz_convert("UTC")
    if timestamp_mode == "fixed_offset":
        if fixed_utc_offset_hours is None:
            raise ScadaDataError(
                "fixed_offset mode requires SCADA_FIXED_UTC_OFFSET_HOURS"
            )
        return parsed.dt.tz_localize("UTC") - pd.Timedelta(
            fixed_utc_offset_hours, unit="h"
        )
    raise ScadaDataError(
        "SCADA timestamp mode must be 'civil_time' or 'fixed_offset', "
        f"not {timestamp_mode!r}"
    )


def load_scada_file(
    path: Path,
    turbine_id: str,
    timestamp_mode: str = SCADA_TIMESTAMP_MODE,
    fixed_utc_offset_hours: float | None = SCADA_FIXED_UTC_OFFSET_HOURS,
) -> pd.DataFrame:
    """Read an original CSV into a validated, clean, UTC-indexable frame."""
    raw = pd.read_csv(path, encoding="utf-8")
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ScadaDataError(f"{path.name} is missing expected columns: {sorted(missing)}")

    frame = raw.loc[:, list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).copy()
    frame["timestamp"] = _as_utc(
        frame["timestamp"], timestamp_mode, fixed_utc_offset_hours
    )
    for name in ("wind_speed", "power", "temperature"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame["turbine_id"] = turbine_id
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if frame["timestamp"].duplicated().any():
        raise ScadaDataError(f"{path.name} contains duplicate timestamps")
    if not frame["power"].between(0, 1).all():
        raise ScadaDataError(f"{path.name} has normalized power outside [0, 1]")
    return frame


def load_all_scada(
    root: Path = PROJECT_ROOT,
    timestamp_mode: str = SCADA_TIMESTAMP_MODE,
    fixed_utc_offset_hours: float | None = SCADA_FIXED_UTC_OFFSET_HOURS,
) -> pd.DataFrame:
    """Load both original files, retaining turbine identity."""
    return pd.concat(
        [
            load_scada_file(
                resolve_turbine_path(turbine_id, root),
                turbine_id,
                timestamp_mode,
                fixed_utc_offset_hours,
            )
            for turbine_id in TURBINES
        ],
        ignore_index=True,
    )


def aggregate_hourly(scada: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to hours and retain coverage diagnostics; never interpolate."""
    required = {"timestamp", "wind_speed", "power", "temperature", "turbine_id"}
    missing = required.difference(scada.columns)
    if missing:
        raise ScadaDataError(f"Cannot aggregate missing columns: {sorted(missing)}")

    hourly_frames: list[pd.DataFrame] = []
    for turbine_id, turbine_data in scada.groupby("turbine_id", sort=True):
        if turbine_data["timestamp"].duplicated().any():
            raise ScadaDataError(f"{turbine_id} contains duplicate raw timestamps")
        turbine_data = turbine_data.copy()
        minute = turbine_data["timestamp"].dt.minute
        turbine_data["_valid_10min_timestamp"] = minute.mod(10).eq(0) & turbine_data[
            "timestamp"
        ].dt.second.eq(0)
        hourly = (
            turbine_data.set_index("timestamp")
            .sort_index()
            .resample("1h")
            .agg(
                wind_speed=("wind_speed", "mean"),
                temperature=("temperature", "mean"),
                power=("power", "mean"),
                coverage_count=("_valid_10min_timestamp", "sum"),
                unique_timestamp_count=("_valid_10min_timestamp", "count"),
                target_power_count=("power", "count"),
            )
            .reset_index()
        )
        # Exactly six cadence-valid timestamps at minute 00,10,...,50 are needed;
        # ordinary row count is not sufficient for a valid target hour.
        expected = (
            turbine_data.set_index("timestamp")["_valid_10min_timestamp"]
            .resample("1h")
            .apply(
                lambda flags: len(flags) == 6
                and flags.index.equals(
                    pd.date_range(
                        flags.index.min().floor("1h"),
                        periods=6,
                        freq="10min",
                        tz="UTC",
                    )
                )
                and bool(flags.all())
            )
            .rename("has_expected_10min_cadence")
            .reset_index(drop=True)
        )
        hourly["has_expected_10min_cadence"] = expected
        hourly["turbine_id"] = turbine_id
        hourly_frames.append(hourly)

    result = pd.concat(hourly_frames, ignore_index=True)
    result["coverage_count"] = result["coverage_count"].astype(int)
    result["is_complete_hour"] = (
        result["coverage_count"].eq(6)
        & result["unique_timestamp_count"].eq(6)
        & result["target_power_count"].eq(6)
        & result["has_expected_10min_cadence"]
    )
    epsilon = 1e-6
    result["zero_power_high_wind"] = result["wind_speed"].gt(5) & result["power"].le(epsilon)
    result["suspected_unavailability"] = (
        result["wind_speed"].gt(5)
        & result["wind_speed"].lt(22)
        & result["power"].le(epsilon)
    )
    return result.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)


def complete_training_hours(
    hourly: pd.DataFrame, exclude_suspected_unavailability: bool = False
) -> pd.DataFrame:
    """Return complete targets, optionally excluding only suspected unavailable hours."""
    mask = hourly["is_complete_hour"].copy()
    if exclude_suspected_unavailability:
        mask &= ~hourly["suspected_unavailability"]
    return hourly.loc[mask].copy()
