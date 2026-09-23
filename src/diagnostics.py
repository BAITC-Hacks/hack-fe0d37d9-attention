"""Pre-February, post-forecast SCADA/weather diagnostics only."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _comparison_metrics(frame: pd.DataFrame, forecast_column: str, actual_column: str) -> dict[str, float | int]:
    valid = frame[[forecast_column, actual_column]].dropna()
    if valid.empty:
        return {"n": 0, "correlation": np.nan, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    forecast = valid[forecast_column].to_numpy(dtype=float)
    actual = valid[actual_column].to_numpy(dtype=float)
    return {
        "n": len(valid),
        "correlation": float(valid[forecast_column].corr(valid[actual_column]))
        if len(valid) >= 2
        else np.nan,
        "mae": float(mean_absolute_error(actual, forecast)),
        "rmse": float(mean_squared_error(actual, forecast) ** 0.5),
        "bias": float(np.mean(forecast - actual)),
    }


def weather_scada_diagnostics(
    diagnostic_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify ECMWF-vs-SCADA domain shift using only already-realised labels."""
    valid = diagnostic_table.loc[diagnostic_table["actual_scada_power"].notna()].copy()
    wind_columns = ["wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m"]
    rows: list[dict[str, object]] = []
    for turbine_id, group in valid.groupby("turbine_id"):
        for weather_column in wind_columns:
            rows.append(
                {
                    "comparison": f"{weather_column}_vs_scada_wind",
                    "turbine_id": turbine_id,
                    "lead_time_hours": "overall",
                    **_comparison_metrics(group, weather_column, "actual_scada_wind_speed"),
                }
            )
        rows.append(
            {
                "comparison": "temperature_2m_vs_scada_temperature",
                "turbine_id": turbine_id,
                "lead_time_hours": "overall",
                **_comparison_metrics(group, "temperature_2m", "actual_scada_temperature"),
            }
        )
        for lead_hours, lead_group in group.groupby("lead_time_hours"):
            rows.append(
                {
                    "comparison": "wind_speed_100m_vs_scada_wind",
                    "turbine_id": turbine_id,
                    "lead_time_hours": int(lead_hours),
                    **_comparison_metrics(
                        lead_group, "wind_speed_100m", "actual_scada_wind_speed"
                    ),
                }
            )

    distributions: list[dict[str, object]] = []
    for turbine_id, group in valid.groupby("turbine_id"):
        for variable, column in {
            "ecmwf_wind_speed_100m": "wind_speed_100m",
            "actual_scada_wind_speed": "actual_scada_wind_speed",
            "ecmwf_temperature_2m": "temperature_2m",
            "actual_scada_temperature": "actual_scada_temperature",
        }.items():
            values = group[column].dropna()
            distributions.append(
                {
                    "turbine_id": turbine_id,
                    "variable": variable,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "p05": float(values.quantile(0.05)),
                    "p50": float(values.quantile(0.50)),
                    "p95": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(distributions)


def timezone_alignment_metrics(
    candidate_to_table: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare explicit timestamp interpretations; never alter configuration."""
    rows: list[dict[str, object]] = []
    for candidate, table in candidate_to_table.items():
        valid = table.loc[table["actual_scada_power"].notna()]
        for turbine_id, group in valid.groupby("turbine_id"):
            rows.append(
                {
                    "timestamp_interpretation": candidate,
                    "scope": turbine_id,
                    **_comparison_metrics(group, "wind_speed_100m", "actual_scada_wind_speed"),
                }
            )
        rows.append(
            {
                "timestamp_interpretation": candidate,
                "scope": "ALL",
                **_comparison_metrics(valid, "wind_speed_100m", "actual_scada_wind_speed"),
            }
        )
    return pd.DataFrame(rows)


def persist_frame(frame: pd.DataFrame, destination: Path) -> Path:
    """Persist diagnostic outputs outside raw source data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
