"""Deterministic forecasting tools shared by rolling inference and the orchestrator.

This module does not read SCADA or observed weather. Frozen models and archived
Single Runs are its only numerical inputs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, TURBINES
from .development import safe_features
from .evaluation import daily_origins
from .features import add_calendar_features
from .leakage import LeakageError, require_utc, validate_weather_samples
from .training_data import write_json
from .weather import MODEL, PROVIDER, REQUESTED_VARIABLES, OpenMeteoSingleRunsClient, WeatherForecast


@dataclass(frozen=True)
class ForecastContext:
    run_id: str
    forecast_origin_utc: pd.Timestamp
    forecast_origin_local: pd.Timestamp
    turbines: tuple[str, ...]
    horizon_hours: int
    model_version: str
    recalculation: bool = False


@dataclass
class WeatherBatch:
    forecasts: dict[str, WeatherForecast]


@dataclass(frozen=True)
class ForecastResult:
    context: ForecastContext
    path: Path
    summary_path: Path
    rows: int
    summary: dict


def load_frozen_models(root: Path = ARTIFACTS_DIR) -> tuple[dict, dict]:
    manifest=json.loads((root/"models/final/manifest.json").read_text("utf-8"))
    models={}
    for turbine,record in manifest["files"].items():
        path=Path(record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            raise ValueError("Frozen model hash mismatch")
        models[turbine]=joblib.load(path)
    return manifest,models


def create_forecast_context(origin: str | pd.Timestamp, turbines: tuple[str,...], manifest: dict,
                            recalculation: bool=False) -> ForecastContext:
    stamp=pd.Timestamp(origin)
    if stamp.tzinfo is None:
        stamp=stamp.tz_localize("Asia/Almaty",ambiguous="raise",nonexistent="raise")
    utc=stamp.tz_convert("UTC")
    if utc!=utc.floor("h"):
        raise ValueError("Forecast origin must be an exact hour")
    if not turbines or len(set(turbines))!=len(turbines) or not set(turbines).issubset(TURBINES):
        raise ValueError("Turbines must be a unique nonempty subset of T1,T2")
    if utc < pd.Timestamp(manifest["training_cutoff_utc"]):
        raise LeakageError("Frozen model is not eligible at this origin")
    return ForecastContext(uuid4().hex,utc,utc.tz_convert("Asia/Almaty"),turbines,
                           48,manifest["model_version"],recalculation)


def fetch_archived_weather(context: ForecastContext, client: OpenMeteoSingleRunsClient) -> WeatherBatch:
    forecasts={}
    for turbine in context.turbines:
        location=TURBINES[turbine]
        forecasts[turbine]=client.get_forecast(location.latitude,location.longitude,context.forecast_origin_utc,48)
    return WeatherBatch(forecasts)


def validate_weather(context: ForecastContext, batch: WeatherBatch) -> dict:
    if set(batch.forecasts)!=set(context.turbines):
        raise ValueError("Weather turbine coverage mismatch")
    runs={}
    for turbine,forecast in batch.forecasts.items():
        frame=forecast.data
        validate_weather_samples(frame)
        if len(frame)!=48 or sorted(frame.lead_time_hours.tolist())!=list(range(1,49)):
            raise ValueError("Weather horizon must contain exactly leads 1..48")
        if not frame.forecast_origin_utc.eq(context.forecast_origin_utc).all():
            raise LeakageError("Weather origin differs from context")
        if frame.valid_time_utc.duplicated().any():
            raise ValueError("Duplicate weather timestamps")
        values=frame[list(REQUESTED_VARIABLES)].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Missing or nonfinite weather values")
        if (frame[[f"wind_speed_{height}m" for height in (10,80,100,120)]]<0).any().any():
            raise ValueError("Negative wind speed")
        if not frame.wind_direction_100m.between(0,360).all():
            raise ValueError("Invalid meteorological wind direction")
        meta=forecast.metadata
        location=TURBINES[turbine]
        if not np.allclose([meta.latitude,meta.longitude],[location.latitude,location.longitude],rtol=0,atol=1e-6):
            raise ValueError("Weather coordinates differ from turbine context")
        if pd.Timestamp(meta.forecast_origin_utc) != context.forecast_origin_utc:
            raise LeakageError("Weather metadata origin differs from context")
        if meta.provider!=PROVIDER or meta.model!=MODEL or meta.request_parameters.get("models")!=MODEL:
            raise ValueError("Unapproved weather source")
        if meta.request_parameters.get("timezone")!="UTC" or meta.request_parameters.get("wind_speed_unit")!="ms":
            raise ValueError("Weather units/timezone contract failure")
        if meta.request_parameters.get("run") != pd.Timestamp(meta.weather_run_init_utc).strftime("%Y-%m-%dT%H:%M"):
            raise ValueError("Pinned request initialization differs from weather metadata")
        if not frame.weather_run_init_utc.eq(pd.Timestamp(meta.weather_run_init_utc)).all():
            raise ValueError("Weather run metadata mismatch")
        if not frame.weather_available_at_utc.eq(pd.Timestamp(meta.weather_available_at_utc)).all():
            raise ValueError("Weather availability metadata mismatch")
        runs[turbine]=dict(run=meta.weather_run_init_utc,available_at=meta.weather_available_at_utc,
                          raw_sha256=meta.raw_response_sha256,cache_hit=forecast.cache_hit,fallback_steps=meta.fallback_steps)
    return dict(status="passed",runs=runs)


def prepare_forecast_features(context: ForecastContext, batch: WeatherBatch) -> pd.DataFrame:
    # Deliberately uses weather-only rows; target/OOF columns cannot enter inference.
    frames=[]
    for turbine,forecast in batch.forecasts.items():
        frame=add_calendar_features(forecast.data)
        frame["turbine_id"]=turbine
        frame["weather_provider"]=PROVIDER
        frame["wind_ratio_100m_10m"]=frame.wind_speed_100m/frame.wind_speed_10m.clip(lower=.1)
        frame=frame.rename(columns={**{f"wind_speed_{h}m":f"wind_speed_{h}m_ms" for h in (10,80,100,120)},
            "wind_direction_100m":"wind_direction_100m_deg","temperature_2m":"temperature_2m_c",
            "surface_pressure":"surface_pressure_hpa","wind_shear_120m_80m":"wind_shear_120m_80m_ms",
            "day_of_year_sin":"dayofyear_sin","day_of_year_cos":"dayofyear_cos"})
        safe_features(frame)
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def run_power_forecast(context: ForecastContext, features: pd.DataFrame, models: dict) -> pd.DataFrame:
    outputs=[]
    for turbine,group in features.groupby("turbine_id",sort=True):
        result=group.copy()
        values=models[turbine].predict(group)
        if not np.isfinite(values).all():
            raise ValueError("Model produced nonfinite power")
        result["predicted_power"]=np.clip(values,0,1)
        result["model_name"]=models[turbine].name
        outputs.append(result)
    result=pd.concat(outputs,ignore_index=True)
    result["run_id"]=context.run_id
    result["model_version"]=context.model_version
    result["forecast_origin_local"]=context.forecast_origin_local
    result["valid_time_local"]=result.valid_time_utc.dt.tz_convert("Asia/Almaty")
    return result


def validate_power_forecast(context: ForecastContext, forecast: pd.DataFrame) -> dict:
    validate_weather_samples(forecast)
    for column in ("forecast_origin_utc","valid_time_utc","weather_run_init_utc","weather_available_at_utc"):
        require_utc(forecast[column],column)
    if any(col.startswith("target_") or col.startswith("actual_") or col=="y_true" for col in forecast):
        raise LeakageError("Observed targets are forbidden in operational forecasts")
    if set(forecast.turbine_id)!=set(context.turbines):
        raise ValueError("Forecast turbine mismatch")
    if not forecast.forecast_origin_utc.eq(context.forecast_origin_utc).all():
        raise LeakageError("Forecast origin mismatch")
    if not forecast.run_id.eq(context.run_id).all():
        raise ValueError("Forecast run identifier mismatch")
    if forecast.duplicated(["turbine_id","valid_time_utc"]).any():
        raise ValueError("Duplicate power forecast")
    for _,group in forecast.groupby("turbine_id"):
        if sorted(group.lead_time_hours.tolist())!=list(range(1,49)):
            raise ValueError("Forecast is not exactly 48 hourly leads")
    if not np.isfinite(forecast.predicted_power).all() or not forecast.predicted_power.between(0,1).all():
        raise ValueError("Forecast power must be finite and within [0,1]")
    return dict(status="passed",rows=len(forecast),horizon_hours=48,observations_used=False)


def summarize_forecast_metrics(context: ForecastContext, forecast: pd.DataFrame) -> dict:
    summaries={}
    for turbine,group in forecast.groupby("turbine_id"):
        group=group.sort_values("lead_time_hours").reset_index(drop=True)
        peak=group.loc[group.predicted_power.idxmax()]
        low=group.loc[group.predicted_power.idxmin()]
        changes=group.predicted_power.diff()
        ramps=[]
        for label,pos in (("ramp_up",changes.idxmax()),("ramp_down",changes.idxmin())):
            ramps.append(dict(kind=label,from_local=group.loc[pos-1,"valid_time_local"].isoformat(),
                to_local=group.loc[pos,"valid_time_local"].isoformat(),change_normalized_power=float(changes.loc[pos])))
        low3=group.predicted_power.rolling(3,min_periods=3).mean()
        pos=low3.idxmin()
        summaries[turbine]=dict(mean_power_24h=float(group.loc[group.lead_time_hours.le(24),"predicted_power"].mean()),
            mean_power_48h=float(group.predicted_power.mean()),peak_power=float(peak.predicted_power),peak_time_local=peak.valid_time_local.isoformat(),
            minimum_power=float(low.predicted_power),minimum_time_local=low.valid_time_local.isoformat(),
            lowest_3h_period=dict(start_local=group.loc[pos-2,"valid_time_local"].isoformat(),end_local=group.loc[pos,"valid_time_local"].isoformat(),mean_power=float(low3.loc[pos])),
            ramps=ramps,wind100_mean_ms=float(group.wind_speed_100m_ms.mean()),
            wind100_min_ms=float(group.wind_speed_100m_ms.min()),wind100_max_ms=float(group.wind_speed_100m_ms.max()),
            temperature_mean_c=float(group.temperature_2m_c.mean()))
    difference={}
    if "T1" in summaries and "T2" in summaries:
        difference={key: summaries["T1"][key]-summaries["T2"][key] for key in ("mean_power_24h","mean_power_48h")}
    return dict(run_id=context.run_id,forecast_origin_utc=context.forecast_origin_utc.isoformat(),
        forecast_origin_local=context.forecast_origin_local.isoformat(),model_name=str(forecast.model_name.iloc[0]),
        horizon_hours=48,turbines=summaries,T1_minus_T2=difference,recalculation=context.recalculation,
        warnings=["Hub height unknown; 100m is a proxy.","Timezone convention remains unconfirmed.",
                  "Seven-hour weather availability is an assumption, not measured publication latency.",
                  "Selected curve underpredicted high generation in pre-February validation.",
                  "No calibrated uncertainty interval; 25-48h validation error exceeded 1-24h error."])


def save_forecast(context: ForecastContext, forecast: pd.DataFrame, summary: dict, root: Path=ARTIFACTS_DIR) -> ForecastResult:
    validate_power_forecast(context,forecast)
    directory=root/"predictions/runs"/context.run_id
    directory.mkdir(parents=True,exist_ok=False)
    path=directory/"forecast.csv"
    forecast.to_csv(path,index=False)
    summary_path=directory/"summary.json"
    write_json(summary,summary_path)
    return ForecastResult(context,path,summary_path,len(forecast),summary)


def forecast_cycle(origin: str | pd.Timestamp, turbines: tuple[str,...], manifest: dict, models: dict,
                   client: OpenMeteoSingleRunsClient, root: Path=ARTIFACTS_DIR) -> ForecastResult:
    context=create_forecast_context(origin,turbines,manifest)
    batch=fetch_archived_weather(context,client)
    validate_weather(context,batch)
    features=prepare_forecast_features(context,batch)
    forecast=run_power_forecast(context,features,models)
    validate_power_forecast(context,forecast)
    summary=summarize_forecast_metrics(context,forecast)
    result=save_forecast(context,forecast,summary,root)
    write_json({t:asdict(f.metadata) for t,f in batch.forecasts.items()},result.path.parent/"weather_provenance.json")
    return result


def persist_february_batch(results: list[ForecastResult], root: Path=ARTIFACTS_DIR, prefix: str="february") -> dict:
    frames=[pd.read_csv(result.path) for result in results]
    forecast=pd.concat(frames,ignore_index=True)
    for column in ("forecast_origin_utc","valid_time_utc","weather_run_init_utc","weather_available_at_utc"):
        forecast[column]=require_utc(forecast[column],column)
    for result in results:
        validate_power_forecast(result.context,forecast.loc[forecast.run_id.eq(result.context.run_id)])
    origins=daily_origins("2026-01-31","2026-02-28")
    if set(forecast.forecast_origin_utc)!=set(origins) or len(forecast)!=29*96:
        raise ValueError("February batch does not contain the full configured origins/turbines")
    destination=root/"predictions"/f"{prefix}_rolling_forecast.csv"
    # Per-run immutable copies remain preserved even when updating a canonical export.
    forecast.to_csv(destination,index=False)
    local_times=forecast.valid_time_utc.dt.tz_convert("Asia/Almaty")
    february=forecast.loc[local_times.ge(pd.Timestamp("2026-02-01",tz="Asia/Almaty")) & local_times.lt(pd.Timestamp("2026-03-01",tz="Asia/Almaty"))]
    submission=february[["turbine_id","forecast_origin_utc","valid_time_utc","lead_time_hours","predicted_power","model_name","run_id"]]
    submission.to_csv(root/"predictions"/f"{prefix}_submission.csv",index=False)
    qc=dict(status="passed",forecast_origins=len(origins),prediction_rows=len(forecast),submission_rows=len(submission),
        first_origin_utc=origins.min().isoformat(),last_origin_utc=origins.max().isoformat(),
        first_origin_local=origins.min().tz_convert("Asia/Almaty").isoformat(),last_origin_local=origins.max().tz_convert("Asia/Almaty").isoformat(),
        full_48h_turbine_folds=58,all_finite=True,all_in_bounds=True,all_weather_available=True,
        observations_endpoint_used=False,february_actual_targets_used=False,model_versions=forecast.model_version.unique().tolist(),
        canonical_output=str(destination),run_ids=[result.context.run_id for result in results])
    write_json(qc,root/"diagnostics"/f"{prefix}_forecast_qc.json")
    write_json({"origins":[result.summary for result in results]},root/"diagnostics"/f"{prefix}_forecast_summaries.json")
    print(json.dumps({key:value for key,value in qc.items() if key!="run_ids"},indent=2),flush=True)
    return qc


def generate_february(client: OpenMeteoSingleRunsClient | None=None) -> dict:
    manifest,models=load_frozen_models()
    client=client or OpenMeteoSingleRunsClient()
    if client.availability_lag_hours != manifest["availability_lag_hours"]:
        raise ValueError("Weather availability policy differs from frozen model manifest")
    results=[]
    for origin in daily_origins("2026-01-31","2026-02-28"):
        result=forecast_cycle(origin,("T1","T2"),manifest,models,client)
        results.append(result)
        print(f"February origin={origin.isoformat()} saved={result.rows} rows run_id={result.context.run_id}",flush=True)
    return persist_february_batch(results)
