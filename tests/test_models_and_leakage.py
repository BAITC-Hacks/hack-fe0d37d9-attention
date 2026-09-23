import numpy as np
import pandas as pd

from src.evaluation import leakage_safe_training_rows
from src.models import clip_predictions


def test_predictions_are_clipped_to_normalized_power_bounds() -> None:
    assert np.array_equal(clip_predictions(np.array([-0.1, 0.3, 1.5])), np.array([0.0, 0.3, 1.0]))


def test_training_cutoff_excludes_future_origin_and_future_target() -> None:
    frame = pd.DataFrame(
        {
            "forecast_origin": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-02T00:00Z", "2026-01-01T00:00Z"], utc=True
            ),
            "target_timestamp": pd.to_datetime(
                ["2026-01-01T12:00Z", "2026-01-01T12:00Z", "2026-01-02T03:00Z"], utc=True
            ),
            "power": [0.1, 0.2, 0.3],
        }
    )
    rows = leakage_safe_training_rows(frame, "2026-01-02T00:00Z")
    assert len(rows) == 1
    assert rows.iloc[0]["power"] == 0.1
