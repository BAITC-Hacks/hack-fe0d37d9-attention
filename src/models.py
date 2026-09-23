"""Transparent empirical and gradient-boosting normalized-power models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import MODEL_FEATURES


def clip_predictions(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


class EmpiricalPowerCurve:
    """Turbine-specific, transparent binned mean power curve."""

    def __init__(self, bin_width: float = 0.5, max_wind_speed: float = 30.0) -> None:
        self.bin_width = bin_width
        self.max_wind_speed = max_wind_speed
        self._curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._fallback: dict[str, float] = {}

    def fit(self, hourly_scada: pd.DataFrame) -> "EmpiricalPowerCurve":
        data = hourly_scada.dropna(subset=["wind_speed", "power", "turbine_id"])
        if "suspected_unavailability" in data:
            data = data.loc[~data["suspected_unavailability"]]
        if data.empty:
            raise ValueError("Cannot fit an empirical curve with no complete SCADA data")
        edges = np.arange(0, self.max_wind_speed + self.bin_width, self.bin_width)
        for turbine_id, group in data.groupby("turbine_id"):
            bins = pd.cut(group["wind_speed"], bins=edges, include_lowest=True)
            grouped = group.groupby(bins, observed=True)["power"].mean()
            centers = np.array([(interval.left + interval.right) / 2 for interval in grouped.index])
            self._curves[str(turbine_id)] = (centers, grouped.to_numpy(dtype=float))
            self._fallback[str(turbine_id)] = float(group["power"].mean())
        return self

    def predict(self, frame: pd.DataFrame, wind_column: str) -> np.ndarray:
        if wind_column not in frame:
            raise ValueError(f"Missing prediction wind column: {wind_column}")
        predictions = np.empty(len(frame), dtype=float)
        for turbine_id, positions in frame.groupby("turbine_id", sort=False).groups.items():
            if str(turbine_id) not in self._curves:
                raise ValueError(f"No empirical curve fitted for turbine {turbine_id}")
            centers, values = self._curves[str(turbine_id)]
            winds = frame.loc[positions, wind_column].to_numpy(dtype=float)
            predictions[frame.index.get_indexer(positions)] = np.interp(
                winds, centers, values, left=values[0], right=values[-1]
            )
        return clip_predictions(predictions)


def chronological_oof_power_curve_predictions(
    rows: pd.DataFrame,
    wind_column: str,
    time_column: str = "valid_time_utc",
    block_size: int = 168,
    min_history_rows: int = 168,
) -> pd.Series:
    """Generate expanding-window OOF curve predictions without target self-use.

    Rows in a chronological block are predicted by a curve fitted only on rows
    with a strictly earlier valid time. Warm-up rows intentionally remain NaN.
    The returned feature is optional; it is not used by the primary baseline.
    """
    if block_size < 1 or min_history_rows < 1:
        raise ValueError("block_size and min_history_rows must be positive")
    required = {"turbine_id", "power", wind_column, time_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"OOF curve rows missing: {sorted(missing)}")
    ordered = rows.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True, errors="raise")
    ordered = ordered.sort_values(time_column)
    output = pd.Series(np.nan, index=rows.index, dtype=float)
    unique_times = ordered[time_column].drop_duplicates().to_list()
    for start in range(0, len(unique_times), block_size):
        block_times = unique_times[start : start + block_size]
        block_start = block_times[0]
        history = ordered.loc[ordered[time_column].lt(block_start)]
        if len(history) < min_history_rows:
            continue
        curve_training = history.rename(columns={wind_column: "wind_speed"})
        curve = EmpiricalPowerCurve().fit(curve_training)
        block = ordered.loc[ordered[time_column].isin(block_times)]
        prediction_input = block.rename(columns={wind_column: "oof_curve_wind_speed"})
        output.loc[block.index] = curve.predict(prediction_input, "oof_curve_wind_speed")
    return output.reindex(rows.index)


def chronological_oof_wind_calibration(
    rows: pd.DataFrame,
    forecast_wind_column: str,
    observed_wind_column: str,
    time_column: str = "valid_time_utc",
    block_size: int = 168,
    min_history_rows: int = 168,
) -> pd.Series:
    """Estimate SCADA wind from forecast wind using strictly earlier target times.

    This is a transparent per-turbine affine calibration intended only as an
    OOF diagnostic feature. Each target-time block is estimated from historical
    archived forecasts and realised SCADA wind strictly before that block.
    """
    if block_size < 1 or min_history_rows < 2:
        raise ValueError("block_size must be positive and min_history_rows at least two")
    required = {"turbine_id", forecast_wind_column, observed_wind_column, time_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Wind-calibration rows missing: {sorted(missing)}")
    ordered = rows.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True, errors="raise")
    ordered = ordered.sort_values(time_column)
    output = pd.Series(np.nan, index=rows.index, dtype=float)
    unique_times = ordered[time_column].drop_duplicates().to_list()
    for start in range(0, len(unique_times), block_size):
        block_times = unique_times[start : start + block_size]
        block_start = block_times[0]
        history = ordered.loc[ordered[time_column].lt(block_start)]
        block = ordered.loc[ordered[time_column].isin(block_times)]
        for turbine_id, block_group in block.groupby("turbine_id", sort=False):
            turbine_history = history.loc[
                history["turbine_id"].eq(turbine_id),
                [forecast_wind_column, observed_wind_column],
            ].dropna()
            if len(turbine_history) < min_history_rows:
                continue
            source = turbine_history[forecast_wind_column].to_numpy(dtype=float)
            target = turbine_history[observed_wind_column].to_numpy(dtype=float)
            if np.isclose(np.std(source), 0.0):
                continue
            slope, intercept = np.polyfit(source, target, deg=1)
            input_wind = block_group[forecast_wind_column].to_numpy(dtype=float)
            output.loc[block_group.index] = np.maximum(0.0, intercept + slope * input_wind)
    return output.reindex(rows.index)


def chronological_oof_cross_domain_power_curve_predictions(
    rows: pd.DataFrame,
    curve_wind_column: str,
    prediction_wind_column: str,
    time_column: str = "valid_time_utc",
    block_size: int = 168,
    min_history_rows: int = 168,
) -> pd.Series:
    """OOF SCADA-wind power curve applied to a forecast-domain wind input.

    The curve itself is fit on earlier measured turbine wind and realised power;
    the next block receives only the supplied prediction-time wind column.
    """
    if block_size < 1 or min_history_rows < 1:
        raise ValueError("block_size and min_history_rows must be positive")
    required = {"turbine_id", "power", curve_wind_column, prediction_wind_column, time_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Cross-domain curve rows missing: {sorted(missing)}")
    ordered = rows.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True, errors="raise")
    ordered = ordered.sort_values(time_column)
    output = pd.Series(np.nan, index=rows.index, dtype=float)
    unique_times = ordered[time_column].drop_duplicates().to_list()
    for start in range(0, len(unique_times), block_size):
        block_times = unique_times[start : start + block_size]
        block_start = block_times[0]
        history_columns = ["turbine_id", "power", curve_wind_column]
        if "suspected_unavailability" in ordered:
            history_columns.append("suspected_unavailability")
        history = ordered.loc[
            ordered[time_column].lt(block_start), history_columns
        ].dropna(subset=["turbine_id", "power", curve_wind_column])
        if len(history) < min_history_rows:
            continue
        curve_training = history.rename(columns={curve_wind_column: "wind_speed"})
        curve = EmpiricalPowerCurve().fit(curve_training)
        block = ordered.loc[ordered[time_column].isin(block_times)]
        available_turbines = set(history["turbine_id"].astype(str))
        block = block.loc[block["turbine_id"].astype(str).isin(available_turbines)]
        if block.empty:
            continue
        prediction_input = block.rename(columns={prediction_wind_column: "power_curve_input"})
        output.loc[block.index] = curve.predict(prediction_input, "power_curve_input")
    return output.reindex(rows.index)


class CatBoostPowerModel:
    """CatBoost regressor using only archived-weather and calendar features."""

    def __init__(self, random_seed: int = 42, iterations: int = 250) -> None:
        self.feature_columns = MODEL_FEATURES
        self.model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="RMSE",
            iterations=iterations,
            depth=7,
            learning_rate=0.06,
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        self.is_fitted = False

    def _feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        features = frame.loc[:, self.feature_columns].copy()
        features["turbine_id"] = features["turbine_id"].astype(str)
        if features.isna().any().any():
            raise ValueError("Model features contain missing values")
        return features

    def fit(self, training_frame: pd.DataFrame) -> "CatBoostPowerModel":
        if "power" not in training_frame:
            raise ValueError("Training frame must contain power")
        features = self._feature_matrix(training_frame)
        target = training_frame["power"].astype(float)
        if target.empty:
            raise ValueError("Cannot fit CatBoost model with no leakage-safe rows")
        self.model.fit(features, target, cat_features=["turbine_id"])
        self.is_fitted = True
        return self

    def predict(self, prediction_frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("CatBoost model is not fitted")
        return clip_predictions(self.model.predict(self._feature_matrix(prediction_frame)))


class HistGradientBoostingTargetModel:
    """Lightweight gradient boosting on the same safe archived-weather features."""

    def __init__(self, target_column: str, max_iter: int = 250) -> None:
        self.target_column = target_column
        self.feature_columns = MODEL_FEATURES
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=max_iter,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            random_state=42,
        )
        self.is_fitted = False

    def _feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        features = frame.loc[:, self.feature_columns].copy()
        features["turbine_id"] = features["turbine_id"].map({"T1": 0.0, "T2": 1.0})
        if features["turbine_id"].isna().any():
            raise ValueError("HistGradientBoosting encountered an unknown turbine_id")
        if features.isna().any().any():
            raise ValueError("Model features contain missing values")
        return features.astype(float)

    def fit(self, training_frame: pd.DataFrame) -> "HistGradientBoostingTargetModel":
        if self.target_column not in training_frame:
            raise ValueError(f"Training frame must contain {self.target_column}")
        target = training_frame[self.target_column].astype(float)
        if target.empty:
            raise ValueError("Cannot fit HistGradientBoosting with no rows")
        self.model.fit(self._feature_matrix(training_frame), target)
        self.is_fitted = True
        return self

    def predict(self, prediction_frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("HistGradientBoosting model is not fitted")
        return self.model.predict(self._feature_matrix(prediction_frame))


class TwoStageWeatherToPowerModel:
    """Forecast weather -> turbine wind -> transparent turbine power curve."""

    def __init__(self) -> None:
        self.wind_model = HistGradientBoostingTargetModel("actual_scada_wind_speed")
        self.power_curve = EmpiricalPowerCurve()
        self.is_fitted = False

    def fit(
        self, forecast_training_rows: pd.DataFrame, prior_complete_scada: pd.DataFrame
    ) -> "TwoStageWeatherToPowerModel":
        self.wind_model.fit(forecast_training_rows)
        self.power_curve.fit(prior_complete_scada)
        self.is_fitted = True
        return self

    def predict(self, prediction_frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Two-stage model is not fitted")
        frame = prediction_frame.copy()
        frame["estimated_turbine_wind_speed"] = self.wind_model.predict(frame)
        return self.power_curve.predict(frame, "estimated_turbine_wind_speed")
