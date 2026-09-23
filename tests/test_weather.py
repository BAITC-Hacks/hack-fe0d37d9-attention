import pandas as pd
import pytest

from src.weather import (
    EXPECTED_HOURLY_UNITS,
    REQUESTED_VARIABLES,
    OpenMeteoSingleRunsClient,
    WeatherArchiveError,
    select_latest_available_run,
)


def test_run_selection_respects_conservative_availability_cutoff() -> None:
    run, cutoff = select_latest_available_run("2026-01-31T00:00:00Z", 7)
    assert cutoff == pd.Timestamp("2026-01-30T17:00:00Z")
    assert run == pd.Timestamp("2026-01-30T12:00:00Z")
    assert run <= cutoff


def test_run_selection_rejects_negative_lag() -> None:
    with pytest.raises(ValueError):
        select_latest_available_run("2026-01-31T00:00:00Z", -1)


def test_weather_request_pins_units_and_utc_unix_time() -> None:
    client = OpenMeteoSingleRunsClient()
    params = client._request_parameters(
        43.64515,
        78.535604,
        pd.Timestamp("2026-01-31T00:00Z"),
        pd.Timestamp("2026-01-30T12:00Z"),
        48,
    )
    assert params["wind_speed_unit"] == "ms"
    assert params["timezone"] == "UTC"
    assert params["timeformat"] == "unixtime"


def test_weather_response_units_are_validated_before_schema_labels_are_applied() -> None:
    payload = {
        "hourly_units": EXPECTED_HOURLY_UNITS.copy(),
        "hourly": {"time": [1_767_225_600], **{name: [1.0] for name in REQUESTED_VARIABLES}},
    }
    parsed = OpenMeteoSingleRunsClient._parse_hourly(payload)
    assert parsed.loc[0, "valid_time_utc"] == pd.Timestamp("2026-01-01T00:00:00Z")
    payload["hourly_units"]["surface_pressure"] = "inHg"
    with pytest.raises(WeatherArchiveError, match="units"):
        OpenMeteoSingleRunsClient._parse_hourly(payload)
