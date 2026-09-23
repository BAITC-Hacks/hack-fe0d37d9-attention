"""Read-only reconciliation of training weather with hashed Single Runs caches."""
from __future__ import annotations

import numpy as np

from .config import ARTIFACTS_DIR, TURBINES
from .development import load_dataset
from .training_data import write_json
from .weather import OpenMeteoSingleRunsClient


class NoNetworkSession:
    def get(self, *args, **kwargs):
        raise RuntimeError("Missing cache: archive audit never downloads replacement weather")


def verify_training_archive() -> dict:
    rows=load_dataset()
    client=OpenMeteoSingleRunsClient(session=NoNetworkSession())
    columns={"wind_speed_10m":"wind_speed_10m_ms", "wind_speed_80m":"wind_speed_80m_ms",
             "wind_speed_100m":"wind_speed_100m_ms","wind_speed_120m":"wind_speed_120m_ms",
             "wind_direction_100m":"wind_direction_100m_deg","temperature_2m":"temperature_2m_c",
             "surface_pressure":"surface_pressure_hpa"}
    records=[]
    for (origin,turbine),block in rows.groupby(["forecast_origin_utc","turbine_id"]):
        location=TURBINES[turbine]
        archive=client.get_forecast(location.latitude,location.longitude,origin)
        joined=block.merge(archive.data,on="valid_time_utc",suffixes=("_dataset","_cache"),validate="one_to_one")
        if len(joined)!=48 or not archive.cache_hit:
            raise ValueError("Dataset archive reconciliation has incomplete horizon")
        for col in ("weather_run_init_utc","weather_available_at_utc","forecast_origin_utc"):
            if not joined[f"{col}_dataset"].eq(joined[f"{col}_cache"]).all():
                raise ValueError(f"Dataset {col} does not match cached archive")
        for raw,target in columns.items():
            np.testing.assert_allclose(joined[raw],joined[target],rtol=1e-12,atol=1e-12)
        records.append(dict(origin=origin.isoformat(),turbine_id=turbine,cache_key=archive.metadata.cache_key,
                            raw_response_sha256=archive.metadata.raw_response_sha256))
    report=dict(status="passed",archive_requests_verified=len(records),rows_verified=len(rows),
                observations_endpoint_used=False,network_requests=0,records=records)
    write_json(report,ARTIFACTS_DIR/"diagnostics/training_archive_reconciliation.json")
    print(f"Training archive reconciliation: {len(records)} hash-checked cached requests, {len(rows)} exact matches")
    return report


if __name__=="__main__":
    verify_training_archive()
