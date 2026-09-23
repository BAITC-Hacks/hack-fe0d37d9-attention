import pandas as pd
import pytest

from src.weather import select_latest_available_run


def test_run_selection_respects_conservative_availability_cutoff() -> None:
    run, cutoff = select_latest_available_run("2026-01-31T00:00:00Z", 7)
    assert cutoff == pd.Timestamp("2026-01-30T17:00:00Z")
    assert run == pd.Timestamp("2026-01-30T12:00:00Z")
    assert run <= cutoff


def test_run_selection_rejects_negative_lag() -> None:
    with pytest.raises(ValueError):
        select_latest_available_run("2026-01-31T00:00:00Z", -1)
