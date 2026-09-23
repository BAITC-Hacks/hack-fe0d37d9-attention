"""SCADA loading and safe hourly aggregation without gap interpolation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, SCADA_TIMEZONE, TURBINES

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


def _as_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(SCADA_TIMEZONE).dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


def load_scada_file(path: Path, turbine_id: str) -> pd.DataFrame:
    """Read an original CSV into a validated, clean, UTC-indexable frame."""
    raw = pd.read_csv(path, encoding="utf-8")
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ScadaDataError(f"{path.name} is missing expected columns: {sorted(missing)}")

    frame = raw.loc[:, list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).copy()
    frame["timestamp"] = _as_utc(frame["timestamp"])
    for name in ("wind_speed", "power", "temperature"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame["turbine_id"] = turbine_id
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if frame["timestamp"].duplicated().any():
        raise ScadaDataError(f"{path.name} contains duplicate timestamps")
    if not frame["power"].between(0, 1).all():
        raise ScadaDataError(f"{path.name} has normalized power outside [0, 1]")
    return frame


def load_all_scada(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load both original files, retaining turbine identity."""
    return pd.concat(
        [
            load_scada_file(resolve_turbine_path(turbine_id, root), turbine_id)
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
        hourly = (
            turbine_data.set_index("timestamp")
            .sort_index()
            .resample("1h")
            .agg(
                wind_speed=("wind_speed", "mean"),
                temperature=("temperature", "mean"),
                power=("power", "mean"),
                coverage_count=("power", "count"),
            )
            .reset_index()
        )
        hourly["turbine_id"] = turbine_id
        hourly_frames.append(hourly)

    result = pd.concat(hourly_frames, ignore_index=True)
    result["is_complete_hour"] = result["coverage_count"].eq(6)
    return result.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)


def complete_training_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """Return only complete target hours; incomplete rows remain available elsewhere."""
    return hourly.loc[hourly["is_complete_hour"]].copy()
