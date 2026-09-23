import numpy as np
import pandas as pd
import pytest

from src.development import origin_oof_curve, fit_curve, ForecastStrategy, STRATEGIES, compare_metrics
from src.leakage import LeakageError, TemporalLeakageGuard, validate_weather_samples


def test_origin_oof_excludes_labels_after_issue_time_even_if_before_target_block():
    origins=pd.date_range("2025-10-01",periods=4,freq="1d",tz="UTC")
    rows=pd.DataFrame([dict(turbine_id="T1",forecast_origin_utc=o,valid_time_utc=o+pd.Timedelta(hours=h),
        target_scada_wind_speed=6.,target_power=.2,is_complete_hour=True,suspected_unavailability=False,
        wind_speed_100m_ms=6.) for o in origins for h in range(1,49)])
    original=origin_oof_curve(rows,min_hours=5)
    changed=rows.copy()
    changed.loc[changed.valid_time_utc.ge(origins[2]),"target_power"]=.9
    altered=origin_oof_curve(changed,min_hours=5)
    block=rows.forecast_origin_utc.eq(origins[2])
    np.testing.assert_allclose(original.loc[block,"power_curve_pred"],altered.loc[block,"power_curve_pred"])
    assert original.loc[block,"curve_fit_max_valid_time_utc"].lt(origins[2]).all()


def test_curve_counts_realised_hour_once():
    rows=pd.DataFrame(dict(turbine_id=["T1"]*3,valid_time_utc=pd.to_datetime(["2025-10-01Z".replace("Z","T00:00Z"),"2025-10-01T00:00Z","2025-10-01T01:00Z"]),
                           target_scada_wind_speed=[6.,6.,6.],target_power=[.2,.2,.8],suspected_unavailability=False))
    prediction=fit_curve(rows).predict(pd.DataFrame(dict(turbine_id=["T1"],wind=[6.])),"wind")
    assert prediction[0]==pytest.approx(.5)


def test_guard_rejects_naive_missing_and_wrong_lead():
    row=pd.DataFrame(dict(forecast_origin_utc=["2026-01-01T00:00Z"],valid_time_utc=["2026-01-01T01:00Z"],
        weather_run_init_utc=["2025-12-31T12:00Z"],weather_available_at_utc=["2025-12-31T19:00Z"],availability_lag_hours=[7],lead_time_hours=[1]))
    validate_weather_samples(row)
    for column,value in [("valid_time_utc",None),("forecast_origin_utc","2026-01-01 00:00"),("lead_time_hours",2)]:
        bad=row.copy(); bad[column]=value
        with pytest.raises(LeakageError): validate_weather_samples(bad)


def test_hourly_label_must_have_finished_at_origin():
    rows=pd.DataFrame(dict(valid_time_utc=pd.to_datetime(["2026-01-01T00:00Z"]),forecast_origin_utc=pd.to_datetime(["2025-12-31T00:00Z"])))
    with pytest.raises(LeakageError): TemporalLeakageGuard.validate_training(rows,pd.Timestamp("2026-01-01T00:30Z"))


def test_curve_inference_cannot_read_evaluation_targets():
    training=pd.DataFrame(dict(turbine_id=["T1"]*2,valid_time_utc=pd.to_datetime(["2025-12-01T01:00Z","2025-12-01T02:00Z"]),
        forecast_origin_utc=pd.to_datetime(["2025-12-01T00:00Z"]*2),target_power=[.2,.4],target_scada_wind_speed=[6.,8.],
        is_complete_hour=True,suspected_unavailability=False))
    model=ForecastStrategy(STRATEGIES[0],"T1").fit(training,pd.Timestamp("2025-12-02T00:00Z"))
    future=pd.DataFrame(dict(turbine_id=["T1"],forecast_origin_utc=pd.to_datetime(["2025-12-02T00:00Z"]),wind_speed_100m_ms=[7.]))
    before=model.predict(future)
    future["target_power"]=1.; future["target_scada_wind_speed"]=100.; future["target_scada_temperature"]=-99.
    np.testing.assert_array_equal(before,model.predict(future))


def test_primary_metrics_include_unavailability_and_report_full_folds_separately():
    p=pd.DataFrame(dict(model_name=[STRATEGIES[0]]*2,turbine_id=["T1"]*2,validation_month=["2025-12"]*2,
        target_power=[0.,.4],predicted_power=[.8,.4],is_complete_hour=True,
        suspected_unavailability=[True,False],fold_complete_48h=[False,True],lead_time_hours=[1,25]))
    m=compare_metrics(p)
    all_rows=m[m.month.eq("combined") & m.turbine_id.eq("ALL") & m.lead_group.eq("1-48h")].set_index("population")
    assert all_rows.loc["all_complete_observed","MAE"]==pytest.approx(.4)
    assert all_rows.loc["available_only","MAE"]==0
    assert all_rows.loc["full_48h_folds","N"]==1
