from pathlib import Path

import pandas as pd
import pytest

from src.scada import (
    ScadaDataError,
    aggregate_hourly,
    complete_training_hours,
    load_scada_file,
)


def test_timestamp_parsing_and_hourly_coverage(tmp_path: Path) -> None:
    path = tmp_path / "scada.csv"
    rows = [
        [index, f"2025-01-01 00:{minute:02d}:00", 5.0, 0.2, -3.0]
        for index, minute in enumerate(range(0, 60, 10), start=1)
    ]
    rows += [
        [7, "2025-01-01 01:00:00", 6.0, 0.3, -2.0],
        [8, "2025-01-01 01:10:00", 6.0, 0.3, -2.0],
    ]
    columns = [
        "ID",
        "Статистическое время",
        "Средняя скорость ветра(m/s)",
        "Нормализованная активная мощность",
        "Средняя температура окружающей среды(°C)",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8")

    loaded = load_scada_file(path, "T1")
    hourly = aggregate_hourly(loaded)
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    assert hourly["coverage_count"].tolist() == [6, 2]
    assert hourly["is_complete_hour"].tolist() == [True, False]
    assert len(complete_training_hours(hourly)) == 1


def test_explicit_fixed_source_utc_offset_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "one_row.csv"
    pd.DataFrame(
        [[1, "2025-01-01 00:00:00", 5.0, 0.2, -3.0]],
        columns=[
            "ID",
            "Статистическое время",
            "Средняя скорость ветра(m/s)",
            "Нормализованная активная мощность",
            "Средняя температура окружающей среды(°C)",
        ],
    ).to_csv(path, index=False, encoding="utf-8")
    loaded = load_scada_file(
        path, "T1", timestamp_mode="fixed_offset", fixed_utc_offset_hours=5
    )
    assert loaded.loc[0, "timestamp"] == pd.Timestamp("2024-12-31T19:00:00Z")


def test_civil_almaty_timezone_uses_historical_pre_and_post_march_2024_offsets(tmp_path: Path) -> None:
    path = tmp_path / "civil_time.csv"
    pd.DataFrame(
        [
            [1, "2024-02-29 23:00:00", 5.0, 0.2, -3.0],
            [2, "2024-03-01 01:00:00", 5.0, 0.2, -3.0],
        ],
        columns=[
            "ID",
            "Статистическое время",
            "Средняя скорость ветра(m/s)",
            "Нормализованная активная мощность",
            "Средняя температура окружающей среды(°C)",
        ],
    ).to_csv(path, index=False, encoding="utf-8")
    loaded = load_scada_file(path, "T1", timestamp_mode="civil_time")
    assert loaded["timestamp"].tolist() == [
        pd.Timestamp("2024-02-29T17:00:00Z"),
        pd.Timestamp("2024-02-29T20:00:00Z"),
    ]


def test_hourly_coverage_requires_unique_expected_10_minute_cadence() -> None:
    irregular = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01T00:00Z",
                    "2025-01-01T00:10Z",
                    "2025-01-01T00:20Z",
                    "2025-01-01T00:30Z",
                    "2025-01-01T00:40Z",
                    "2025-01-01T00:55Z",
                ],
                utc=True,
            ),
            "wind_speed": [6.0] * 6,
            "temperature": [0.0] * 6,
            "power": [0.2] * 6,
            "turbine_id": ["T1"] * 6,
        }
    )
    hourly = aggregate_hourly(irregular)
    assert hourly.loc[0, "coverage_count"] == 5
    assert hourly.loc[0, "unique_timestamp_count"] == 6
    assert not hourly.loc[0, "has_expected_10min_cadence"]
    assert not hourly.loc[0, "is_complete_hour"]


def test_duplicate_raw_timestamps_are_rejected_and_unavailability_is_optional_filter() -> None:
    timestamps = pd.date_range("2025-01-01T00:00Z", periods=6, freq="10min")
    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind_speed": [6.0] * 6,
            "temperature": [0.0] * 6,
            "power": [0.0] * 6,
            "turbine_id": ["T1"] * 6,
        }
    )
    hourly = aggregate_hourly(raw)
    assert hourly.loc[0, "zero_power_high_wind"]
    assert hourly.loc[0, "suspected_unavailability"]
    assert len(complete_training_hours(hourly)) == 1
    assert complete_training_hours(hourly, exclude_suspected_unavailability=True).empty

    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ScadaDataError, match="duplicate"):
        aggregate_hourly(duplicate)
