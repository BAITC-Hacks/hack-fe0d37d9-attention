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

# The source SCADA timestamps contain no timezone offset. Phase 1 treats them as
# UTC so they can be joined safely to UTC weather valid times. This is an explicit
# assumption that must be revisited if organisers provide a plant-local timezone.
SCADA_TIMEZONE = os.getenv("SCADA_TIMEZONE", "UTC")
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
