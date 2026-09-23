import numpy as np
import pandas as pd
import pytest

from src.leakage import LeakageError, filter_outer_fold_training_rows, validate_weather_samples
from src.models import (
    EmpiricalPowerCurve,
    chronological_oof_power_curve_predictions,
    clip_predictions,
)
from src.training_data import get_training_rows_before


def test_predictions_are_clipped_to_normalized_power_bounds() -> None:
    assert np.array_equal(clip_predictions(np.array([-0.1, 0.3, 1.5])), np.array([0.0, 0.3, 1.0]))


def test_outer_training_cutoff_rejects_earlier_forecast_with_future_target() -> None:
    frame = pd.DataFrame(
        {
            "forecast_origin_utc": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-02T00:00Z", "2026-01-01T00:00Z"], utc=True
            ),
            "valid_time_utc": pd.to_datetime(
                ["2026-01-01T12:00Z", "2026-01-01T12:00Z", "2026-01-02T03:00Z"], utc=True
            ),
            "power": [0.1, 0.2, 0.3],
        }
    )
    rows = filter_outer_fold_training_rows(frame, "2026-01-02T00:00Z")
    assert len(rows) == 1
    assert rows.iloc[0]["power"] == 0.1


def test_weather_temporal_contract_rejects_unavailable_or_non_future_weather() -> None:
    valid = pd.DataFrame(
        {
            "forecast_origin_utc": ["2026-01-01T00:00Z"],
            "weather_run_init_utc": ["2025-12-31T12:00Z"],
            "weather_available_at_utc": ["2025-12-31T19:00Z"],
            "valid_time_utc": ["2026-01-01T01:00Z"],
            "lead_time_hours": [1],
            "availability_lag_hours": [7],
        }
    )
    validate_weather_samples(valid)
    invalid = valid.copy()
    invalid.loc[0, "weather_available_at_utc"] = "2026-01-01T01:00Z"
    with pytest.raises(LeakageError):
        validate_weather_samples(invalid)


def test_power_curve_is_turbine_specific_and_filters_suspected_unavailability() -> None:
    training = pd.DataFrame(
        {
            "turbine_id": ["T1", "T1", "T2"],
            "wind_speed": [6.0, 6.0, 6.0],
            "power": [0.0, 0.8, 0.2],
            "suspected_unavailability": [True, False, False],
        }
    )
    prediction = pd.DataFrame({"turbine_id": ["T1", "T2"], "wind_speed": [6.0, 6.0]})
    values = EmpiricalPowerCurve().fit(training).predict(prediction, "wind_speed")
    assert np.allclose(values, [0.8, 0.2])


def test_chronological_oof_curve_never_uses_its_own_or_future_block_targets() -> None:
    rows = pd.DataFrame(
        {
            "valid_time_utc": pd.date_range("2025-01-01T00:00Z", periods=8, freq="1h"),
            "turbine_id": ["T1"] * 8,
            "wind_speed_100m": np.arange(1.0, 9.0),
            "power": np.arange(0.1, 0.9, 0.1),
        }
    )
    original = chronological_oof_power_curve_predictions(
        rows, "wind_speed_100m", block_size=2, min_history_rows=4
    )
    altered = rows.copy()
    altered.loc[4, "power"] = 1.0
    changed = chronological_oof_power_curve_predictions(
        altered, "wind_speed_100m", block_size=2, min_history_rows=4
    )
    assert original.iloc[:4].isna().all()
    assert np.allclose(original.iloc[4:6], changed.iloc[4:6], equal_nan=True)


def test_training_dataset_helper_requires_both_temporal_cutoffs_and_target_quality() -> None:
    rows = pd.DataFrame(
        {
            "forecast_origin_utc": pd.to_datetime(
                ["2026-01-01T00:00Z"] * 4, utc=True
            ),
            "valid_time_utc": pd.to_datetime(
                [
                    "2026-01-01T01:00Z",
                    "2026-01-01T02:00Z",
                    "2026-01-01T03:00Z",
                    "2026-01-02T01:00Z",
                ],
                utc=True,
            ),
            "target_power": [0.2, 0.3, 0.4, 0.5],
            "is_complete_hour": [True, False, True, True],
            "suspected_unavailability": [False, False, True, False],
        }
    )
    eligible = get_training_rows_before(rows, "2026-01-02T00:00Z")
    assert eligible["valid_time_utc"].tolist() == [pd.Timestamp("2026-01-01T01:00Z")]
