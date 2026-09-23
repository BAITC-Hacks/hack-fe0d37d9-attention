"""Origin-safe model development. Never reads raw SCADA or February labels."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from .config import ARTIFACTS_DIR
from .evaluation import daily_origins
from .leakage import LeakageError, TemporalLeakageGuard, require_utc, validate_weather_samples
from .models import EmpiricalPowerCurve, clip_predictions
from .training_data import get_training_rows_before, write_json

STRATEGIES = ("A_direct_curve", "B_calibrated_curve", "C_hist_gradient", "D_hybrid_catboost")
FIRST_REPLAY_ORIGIN = pd.Timestamp("2026-01-31", tz="Asia/Almaty").tz_convert("UTC")
CAT_PARAMS = dict(loss_function="RMSE", iterations=450, depth=6, learning_rate=0.05,
                  random_seed=42, verbose=False, allow_writing_files=False, thread_count=2)
HIST_PARAMS = dict(loss="squared_error", max_iter=160, max_leaf_nodes=15,
                   learning_rate=0.05, l2_regularization=3.0, min_samples_leaf=30,
                   early_stopping=False, random_state=42)
SAFE_FEATURES = [
    "wind_speed_10m_ms", "wind_speed_80m_ms", "wind_speed_100m_ms", "wind_speed_120m_ms",
    "wind_shear_120m_80m_ms", "wind_ratio_100m_10m", "wind_ratio_100m_80m",
    "wind_direction_100m_deg", "u100_ms", "v100_ms", "temperature_2m_c", "surface_pressure_hpa",
    "lead_time_hours", "model_lead_time_hours", "run_age_at_origin_hours",
    "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos",
]
TIME_COLUMNS = ["forecast_origin_utc", "weather_run_init_utc", "weather_available_at_utc", "valid_time_utc"]


def safe_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    result["wind_ratio_100m_80m"] = result["wind_speed_100m_ms"] / result["wind_speed_80m_ms"].clip(lower=0.1)
    values = result[SAFE_FEATURES].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite or missing prediction-time features")
    return values


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or ARTIFACTS_DIR / "datasets/forecast_training.csv"
    # Prior Phase 2A target-block OOF columns are intentionally ignored.
    rows = pd.read_csv(path).drop(columns=[
        "calibrated_wind_estimate_oof_ms", "power_curve_direct_oof_pred", "power_curve_calibrated_oof_pred"
    ], errors="ignore")
    for col in TIME_COLUMNS:
        rows[col] = require_utc(rows[col], col)
    for col in ("is_complete_hour", "suspected_unavailability"):
        if not rows[col].isin([True, False]).all():
            raise ValueError(f"Invalid boolean target-quality column {col}")
    if rows.valid_time_utc.ge(pd.Timestamp("2026-02-01", tz="Asia/Almaty")).any():
        raise LeakageError("Dataset contains February labels")
    if rows.duplicated(["turbine_id", "forecast_origin_utc", "valid_time_utc"]).any():
        raise ValueError("Duplicate forecast keys")
    validate_weather_samples(rows)
    safe_features(rows)
    return rows.sort_values(["forecast_origin_utc", "turbine_id", "valid_time_utc"]).reset_index(drop=True)


def fit_curve(training: pd.DataFrame) -> EmpiricalPowerCurve:
    # Each realised SCADA hour counts once, despite overlapping forecast horizons.
    unique = training.drop_duplicates(["turbine_id", "valid_time_utc"])
    return EmpiricalPowerCurve().fit(unique[[
        "turbine_id", "target_scada_wind_speed", "target_power", "suspected_unavailability"
    ]].rename(columns={"target_scada_wind_speed": "wind_speed", "target_power": "power"}))


def origin_oof_curve(rows: pd.DataFrame, min_hours: int = 168) -> pd.DataFrame:
    result = rows.copy()
    result["power_curve_pred"] = np.nan
    result["curve_fit_max_valid_time_utc"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    for origin, block in rows.groupby("forecast_origin_utc", sort=True):
        history = get_training_rows_before(rows, origin)
        for turbine, group in block.groupby("turbine_id"):
            prior = history.loc[history.turbine_id.eq(turbine)]
            if prior.valid_time_utc.nunique() < min_hours:
                continue
            TemporalLeakageGuard.validate_training(prior, origin)
            result.loc[group.index, "power_curve_pred"] = fit_curve(prior).predict(group, "wind_speed_100m_ms")
            result.loc[group.index, "curve_fit_max_valid_time_utc"] = prior.valid_time_utc.max()
    valid = result.power_curve_pred.notna()
    if not result.loc[valid, "curve_fit_max_valid_time_utc"].lt(result.loc[valid, "forecast_origin_utc"]).all():
        raise LeakageError("OOF curve learned from unavailable labels")
    return result


class ForecastStrategy:
    """One reproducible strategy for one turbine; targets are never inference features."""

    def __init__(self, name: str, turbine_id: str) -> None:
        if name not in STRATEGIES:
            raise ValueError(name)
        self.name, self.turbine_id = name, turbine_id
        self.curve = None
        self.regressor = None

    def fit(self, training: pd.DataFrame, origin: pd.Timestamp) -> "ForecastStrategy":
        training = training.loc[training.turbine_id.eq(self.turbine_id)].copy()
        TemporalLeakageGuard.validate_training(training, origin)
        if training.empty or not training.is_complete_hour.all() or training.suspected_unavailability.any():
            raise ValueError("Invalid training target quality")
        self.training_cutoff = origin
        self.training_max_valid_time = training.valid_time_utc.max()
        self.training_rows = len(training)
        self.curve = fit_curve(training)
        if self.name == STRATEGIES[0]:
            return self
        features = safe_features(training)
        if self.name == STRATEGIES[3]:
            if "power_curve_pred" not in training:
                raise LeakageError("Hybrid training requires chronological origin OOF predictions")
            keep = training.power_curve_pred.notna()
            oof = training.loc[keep]
            if not oof.curve_fit_max_valid_time_utc.lt(oof.forecast_origin_utc).all():
                raise LeakageError("Invalid OOF feature provenance")
            features = features.loc[keep].copy()
            features["power_curve_pred"] = oof.power_curve_pred
            target = oof.target_power
            self.regressor = CatBoostRegressor(**CAT_PARAMS)
        else:
            target = training.target_scada_wind_speed if self.name == STRATEGIES[1] else training.target_power
            self.regressor = HistGradientBoostingRegressor(**HIST_PARAMS)
        with threadpool_limits(limits=2):
            self.regressor.fit(features, target)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if not rows.turbine_id.eq(self.turbine_id).all():
            raise ValueError("Turbine model mismatch")
        if self.curve is None:
            raise ValueError("Model has not been fitted")
        if rows.forecast_origin_utc.lt(self.training_cutoff).any():
            raise LeakageError("Model was trained after requested forecast origin")
        if self.name == STRATEGIES[0]:
            return self.curve.predict(rows, "wind_speed_100m_ms")
        features = safe_features(rows)
        if self.name == STRATEGIES[3]:
            features["power_curve_pred"] = self.curve.predict(rows, "wind_speed_100m_ms")
        with threadpool_limits(limits=2):
            values = self.regressor.predict(features)
        if self.name == STRATEGIES[1]:
            predicted = rows.copy()
            predicted["calibrated_wind_ms"] = np.maximum(values, 0)
            return self.curve.predict(predicted, "calibrated_wind_ms")
        return clip_predictions(values)


def metrics(rows: pd.DataFrame) -> dict:
    error = rows.predicted_power - rows.target_power
    return dict(N=len(rows), MAE=float(error.abs().mean()), RMSE=float(np.sqrt((error**2).mean())),
                R2=float(r2_score(rows.target_power, rows.predicted_power)) if len(rows)>1 else None,
                normalized_capacity_error_pp=float(error.abs().mean()*100), bias=float(error.mean()))


def compare_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    observed = predictions.loc[predictions.is_complete_hour & np.isfinite(predictions.target_power)].copy()
    output = []
    for population in ("all_complete_observed", "available_only", "full_48h_folds"):
        population_rows = observed
        if population == "available_only":
            population_rows = observed.loc[~observed.suspected_unavailability]
        if population == "full_48h_folds":
            population_rows = observed.loc[observed.fold_complete_48h]
        for month in ("2025-12", "2026-01", "combined"):
            month_rows = population_rows if month == "combined" else population_rows.loc[population_rows.validation_month.eq(month)]
            for turbine in ("T1", "T2", "ALL"):
                turbine_rows = month_rows if turbine == "ALL" else month_rows.loc[month_rows.turbine_id.eq(turbine)]
                for band, lo, hi in (("1-24h",1,24),("25-48h",25,48),("1-48h",1,48)):
                    for name, group in turbine_rows.loc[turbine_rows.lead_time_hours.between(lo,hi)].groupby("model_name"):
                        output.append(dict(population=population, month=month, turbine_id=turbine,
                                           lead_group=band, model_name=name, **metrics(group)))
    return pd.DataFrame(output)


def error_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    observed = predictions.loc[predictions.is_complete_hour & np.isfinite(predictions.target_power)].copy()
    observed["wind_bucket"] = pd.cut(observed.target_scada_wind_speed, [-np.inf,3,5,8,12,np.inf]).astype(str)
    observed["generation_bucket"] = pd.cut(observed.target_power, [-0.01,0.1,0.7,1.01], labels=["low","medium","high"]).astype(str)
    observed["extreme_generation"] = np.select([observed.target_power.le(.02),observed.target_power.ge(.95)], ["very_low","near_rated"], default="intermediate")
    observed["hour_local"] = observed.valid_time_utc.dt.tz_convert("Asia/Almaty").dt.hour
    observed["lead_group"] = np.where(observed.lead_time_hours.le(24),"1-24h","25-48h")
    result=[]
    for dimension in ("lead_group","lead_time_hours","wind_bucket","generation_bucket","hour_local","validation_month","extreme_generation"):
        for (name,turbine,bucket), group in observed.groupby(["model_name","turbine_id",dimension],observed=True):
            result.append(dict(model_name=name,turbine_id=turbine,dimension=dimension,bucket=str(bucket),**metrics(group)))
    return pd.DataFrame(result)


def run_model_development(start: str="2025-12-01", end: str="2026-01-29", output: Path=ARTIFACTS_DIR) -> dict:
    rows = origin_oof_curve(load_dataset())
    origins = daily_origins(start,end)
    predictions, audits = [], []
    for index, origin in enumerate(origins):
        training = get_training_rows_before(rows, origin)
        TemporalLeakageGuard.validate_training(training, origin)
        for turbine in ("T1", "T2"):
            evaluation = rows.loc[rows.forecast_origin_utc.eq(origin) & rows.turbine_id.eq(turbine)].copy()
            if sorted(evaluation.lead_time_hours.tolist()) != list(range(1,49)):
                raise ValueError(f"Missing full weather horizon for {origin}/{turbine}; refusing partial forecast")
            complete = evaluation.is_complete_hour & np.isfinite(evaluation.target_power)
            evaluation["fold_complete_48h"] = bool(complete.all())
            evaluation["fold_observed_hours"] = int(complete.sum())
            evaluation["validation_month"] = origin.tz_convert("Asia/Almaty").strftime("%Y-%m")
            audits.append(dict(origin=origin.isoformat(),turbine_id=turbine,observed_hours=int(complete.sum()),
                               status="complete" if complete.all() else "partial_explicitly_marked",
                               max_training_target=training.valid_time_utc.max().isoformat(),training_rows=len(training)))
            for name in STRATEGIES:
                model = ForecastStrategy(name,turbine).fit(training,origin)
                result=evaluation.copy()
                result["predicted_power"]=model.predict(evaluation)
                result["model_name"]=name
                if not np.isfinite(result.predicted_power).all() or not result.predicted_power.between(0,1).all():
                    raise ValueError("Prediction bounds failure")
                predictions.append(result)
        print(f"walk-forward {index+1}/{len(origins)} origin={origin.isoformat()}",flush=True)
    forecast=pd.concat(predictions,ignore_index=True)
    comparison=compare_metrics(forecast)
    errors=error_slices(forecast)
    primary=comparison.loc[comparison.population.eq("all_complete_observed") & comparison.month.eq("combined") & comparison.turbine_id.eq("ALL") & comparison.lead_group.eq("1-48h")].sort_values("MAE")
    best=primary.iloc[0]
    # Prespecified simplicity tie tolerance: 0.2 percentage points rated capacity.
    tied=primary.loc[primary.MAE.le(best.MAE+0.002)]
    chosen=min(tied.model_name, key=lambda name: STRATEGIES.index(name))
    daily=forecast.loc[forecast.is_complete_hour].groupby(["model_name","forecast_origin_utc"]).apply(lambda g: (g.predicted_power-g.target_power).abs().mean(),include_groups=False)
    selection=dict(selected_strategy=chosen,selection_metric="held-out all-complete observed MAE",tie_tolerance_mae=0.002,
        complexity_order=list(STRATEGIES),rationale="Minimize held-out pooled MAE; prefer simpler strategy within 0.002. Both months, horizons and full-fold metrics retained for stability review.",
        selected_metrics=comparison.loc[comparison.model_name.eq(chosen)].to_dict("records"),
        daily_mae_summary=daily.groupby(level=0).agg(["mean","std","max"]).reset_index().to_dict("records"),
        validation_origin_start_local=start,validation_origin_end_local=end,
        dataset_sha256=hashlib.sha256((ARTIFACTS_DIR/"datasets/forecast_training.csv").read_bytes()).hexdigest(),
        parameters={"catboost":CAT_PARAMS,"hist_gradient":HIST_PARAMS},
        oof_policy="per forecast origin; strictly earlier realised targets; duplicate SCADA hours removed from curves",
        february_targets_used=False,fold_policy="All complete observed target hours primary; partial target folds explicitly flagged. Full-48h-fold comparison provided separately.",
        gate_status="pending_review",created_at=pd.Timestamp.now(tz="UTC").isoformat())
    for folder in ("metrics","diagnostics","predictions","models"):
        (output/folder).mkdir(parents=True,exist_ok=True)
    forecast.to_csv(output/"predictions/walk_forward_predictions.csv",index=False)
    comparison.to_csv(output/"metrics/model_comparison.csv",index=False)
    write_json({"metrics":comparison.to_dict("records")},output/"metrics/model_comparison.json")
    errors.to_csv(output/"diagnostics/error_analysis.csv",index=False)
    pd.DataFrame(audits).to_csv(output/"diagnostics/outer_fold_audit.csv",index=False)
    rows[["turbine_id","forecast_origin_utc","valid_time_utc","power_curve_pred","curve_fit_max_valid_time_utc"]].to_csv(output/"datasets/origin_oof_curve_features.csv",index=False)
    write_json(selection,output/"models/model_selection.json")
    print(primary.to_string(index=False),flush=True)
    return selection


def freeze_final_models() -> dict:
    selection=json.loads((ARTIFACTS_DIR/"models/model_selection.json").read_text("utf-8"))
    if selection.get("gate_status") != "passed":
        raise ValueError("Phase 2B acceptance gate has not passed")
    dataset_path=ARTIFACTS_DIR/"datasets/forecast_training.csv"
    dataset_hash=hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if selection["dataset_sha256"] != dataset_hash:
        raise ValueError("Dataset changed after model selection")
    rows=origin_oof_curve(load_dataset())
    training=get_training_rows_before(rows,FIRST_REPLAY_ORIGIN)
    TemporalLeakageGuard.validate_training(training,FIRST_REPLAY_ORIGIN)
    model_version=uuid4().hex
    directory=ARTIFACTS_DIR/"models/final"/model_version
    directory.mkdir(parents=True,exist_ok=False)
    files={}
    for turbine in ("T1","T2"):
        model=ForecastStrategy(selection["selected_strategy"],turbine).fit(training,FIRST_REPLAY_ORIGIN)
        path=directory/f"{turbine}.joblib"
        joblib.dump(model,path)
        files[turbine]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest=dict(model_version=model_version,model_name=selection["selected_strategy"],files=files,
        training_cutoff_utc=FIRST_REPLAY_ORIGIN.isoformat(),training_max_valid_time_utc=training.valid_time_utc.max().isoformat(),
        training_rows=len(training),training_rows_per_turbine=training.groupby("turbine_id").size().to_dict(),
        features=SAFE_FEATURES+(["power_curve_pred"] if selection["selected_strategy"]==STRATEGIES[3] else []),
        packages={name:version(name) for name in ("numpy","pandas","scikit-learn","catboost","joblib","requests")},
        random_seed=42,parameters=selection["parameters"],timezone_policy="civil_time/Asia/Almaty",availability_lag_hours=7,
        primary_wind_proxy_m=100,dataset_sha256=dataset_hash,created_at=pd.Timestamp.now(tz="UTC").isoformat())
    write_json(manifest,directory/"manifest.json")
    write_json(manifest,ARTIFACTS_DIR/"models/final/manifest.json")
    print(json.dumps({k:manifest[k] for k in ("model_name","model_version","training_cutoff_utc","training_rows")},indent=2))
    return manifest


def accept_validation(review: str, tests_passed: int) -> dict:
    """Persist an acceptance decision only after deterministic checks and explicit review."""
    if not review.strip() or tests_passed < 1:
        raise ValueError("A validation review and successful test count are required")
    selection_path=ARTIFACTS_DIR/"models/model_selection.json"
    selection=json.loads(selection_path.read_text("utf-8"))
    forecast=pd.read_csv(ARTIFACTS_DIR/"predictions/walk_forward_predictions.csv")
    for col in TIME_COLUMNS:
        forecast[col]=require_utc(forecast[col],col)
    validate_weather_samples(forecast)
    if forecast.valid_time_utc.ge(pd.Timestamp("2026-02-01",tz="Asia/Almaty")).any():
        raise LeakageError("February validation targets")
    if not np.isfinite(forecast.predicted_power).all() or not forecast.predicted_power.between(0,1).all():
        raise ValueError("Invalid validation predictions")
    for _, block in forecast.groupby(["model_name","turbine_id","forecast_origin_utc"]):
        if sorted(block.lead_time_hours.tolist())!=list(range(1,49)):
            raise ValueError("Incomplete weather horizon")
    audit=pd.read_csv(ARTIFACTS_DIR/"diagnostics/outer_fold_audit.csv")
    if not pd.to_datetime(audit.max_training_target,utc=True).lt(pd.to_datetime(audit.origin,utc=True)).all():
        raise LeakageError("Invalid outer-fold training cutoff")
    if set(forecast.validation_month)!={"2025-12","2026-01"}:
        raise ValueError("Both validation months are required")
    oof=pd.read_csv(ARTIFACTS_DIR/"datasets/origin_oof_curve_features.csv").dropna(subset=["power_curve_pred"])
    if not pd.to_datetime(oof.curve_fit_max_valid_time_utc,utc=True).lt(pd.to_datetime(oof.forecast_origin_utc,utc=True)).all():
        raise LeakageError("OOF fit cutoff violation")
    # All candidates must be evaluated on precisely identical row keys/masks.
    keys=["turbine_id","forecast_origin_utc","valid_time_utc","is_complete_hour","suspected_unavailability"]
    reference=None
    for _, group in forecast.groupby("model_name"):
        mask=group[keys].sort_values(keys[:3]).reset_index(drop=True)
        if reference is not None and not reference.equals(mask):
            raise ValueError("Models were evaluated on different target populations")
        reference=mask
    gate=dict(status="passed",tests_passed=tests_passed,review=review,
        bounds=True,weather_available_by_origin=True,training_targets_strictly_before_origin=True,
        oof_cutoffs_valid=True,identical_evaluation_population=True,february_targets_used=False,
        full_turbine_folds=int(audit.status.eq("complete").sum()),partial_turbine_folds=int(audit.status.ne("complete").sum()),
        checked_at=pd.Timestamp.now(tz="UTC").isoformat())
    selection["gate_status"]="passed"
    selection["acceptance_review"]=review
    write_json(gate,ARTIFACTS_DIR/"diagnostics/phase2b_acceptance.json")
    write_json(selection,selection_path)
    return gate
