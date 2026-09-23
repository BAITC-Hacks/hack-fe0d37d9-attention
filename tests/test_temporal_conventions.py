import pandas as pd

from src.evaluation import daily_origins


def test_daily_origins_are_almaty_midnight_converted_to_utc() -> None:
    origins = daily_origins("2026-01-01", "2026-01-02")
    assert origins.tolist() == [
        pd.Timestamp("2025-12-31T19:00:00Z"),
        pd.Timestamp("2026-01-01T19:00:00Z"),
    ]
