"""Project configuration and explicit modelling assumptions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "weather"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# SCADA source timestamps are timezone-naive. The default is civil time using
# Asia/Almaty's historical IANA rules; fixed_offset requires a conscious, named
# configuration. Neither mode claims to prove the organiser's source convention.
SCADA_TIMESTAMP_MODE = os.getenv("SCADA_TIMESTAMP_MODE", "civil_time")
SCADA_CIVIL_TIMEZONE = "Asia/Almaty"
_fixed_offset = os.getenv("SCADA_FIXED_UTC_OFFSET_HOURS")
SCADA_FIXED_UTC_OFFSET_HOURS = float(_fixed_offset) if _fixed_offset else None
FORECAST_ORIGIN_CONVENTION = os.getenv(
    "FORECAST_ORIGIN_CONVENTION", "almaty_midnight"
)
DEFAULT_AVAILABILITY_LAG_HOURS = int(
    os.getenv("WEATHER_AVAILABILITY_LAG_HOURS", "7")
)


@dataclass(frozen=True)
class Turbine:
    turbine_id: str
    latitude: float
    longitude: float
    source_glob: str


TURBINES: dict[str, Turbine] = {
    "T1": Turbine(
        turbine_id="T1",
        latitude=43.645150,
        longitude=78.535604,
        source_glob="*turbine 1.csv",
    ),
    "T2": Turbine(
        turbine_id="T2",
        latitude=43.643198,
        longitude=78.538828,
        source_glob="*turbine 2.csv",
    ),
}

# 100 m is a modelling proxy, not a turbine specification supplied by organisers.
PRIMARY_WIND_FEATURE = "wind_speed_100m"
