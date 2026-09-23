"""Transparent empirical and gradient-boosting normalized-power models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

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
