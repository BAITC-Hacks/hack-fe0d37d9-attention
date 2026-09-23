import pandas as pd

from src.features import MODEL_FEATURES, add_calendar_features


def test_derived_weather_features_use_only_prediction_time_columns() -> None:
    frame = pd.DataFrame(
        {
            "valid_time_utc": ["2026-01-01T04:00:00Z"],
            "forecast_origin_utc": ["2026-01-01T00:00:00Z"],
            "weather_run_init_utc": ["2025-12-31T12:00:00Z"],
            "wind_speed_10m": [3.0],
            "wind_speed_80m": [8.0],
            "wind_speed_100m": [9.0],
            "wind_speed_120m": [10.0],
            "wind_direction_100m": [90.0],
            "temperature_2m": [2.0],
            "surface_pressure": [900.0],
        }
    )
    features = add_calendar_features(frame)
    assert features.loc[0, "wind_shear_120m_80m"] == 2.0
    assert features.loc[0, "wind_ratio_100m_80m"] == 9.0 / 8.0
    assert "actual_scada_wind_speed" not in MODEL_FEATURES
    assert "actual_scada_temperature" not in MODEL_FEATURES
    assert "power" not in MODEL_FEATURES


def test_meteorological_u_v_components_follow_cardinal_direction_convention() -> None:
    directions = [0.0, 90.0, 180.0, 270.0]
    frame = pd.DataFrame(
        {
            "valid_time_utc": ["2026-01-01T01:00:00Z"] * 4,
            "forecast_origin_utc": ["2026-01-01T00:00:00Z"] * 4,
            "weather_run_init_utc": ["2025-12-31T12:00:00Z"] * 4,
            "wind_speed_10m": [10.0] * 4,
            "wind_speed_80m": [10.0] * 4,
            "wind_speed_100m": [10.0] * 4,
            "wind_speed_120m": [10.0] * 4,
            "wind_direction_100m": directions,
            "temperature_2m": [0.0] * 4,
            "surface_pressure": [900.0] * 4,
        }
    )
    features = add_calendar_features(frame)
    assert features["u100_ms"].round(6).tolist() == [0.0, -10.0, 0.0, 10.0]
    assert features["v100_ms"].round(6).tolist() == [-10.0, 0.0, 10.0, 0.0]
