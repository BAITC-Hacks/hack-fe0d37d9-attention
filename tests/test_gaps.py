import pandas as pd

from src.gaps import reindex_hourly_with_segments, segment_safe_rolling_mean, segment_safe_shift


def test_gap_reindex_breaks_lags_and_rolls_between_segments() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T03:00Z"], utc=True
            ),
            "turbine_id": ["T1", "T1", "T1"],
            "power": [0.1, 0.2, 0.4],
            "is_complete_hour": [True, True, True],
        }
    )
    segmented = reindex_hourly_with_segments(hourly)
    assert segmented["timestamp"].tolist() == list(
        pd.date_range("2026-01-01T00:00Z", periods=4, freq="1h")
    )
    assert segmented["segment_id"].astype("string").tolist() == ["1", "1", pd.NA, "2"]
    assert segment_safe_shift(segmented, "power").isna().tolist() == [True, False, True, True]
    assert segment_safe_rolling_mean(segmented, "power", window=2).isna().tolist() == [True, False, True, True]
