"""Synthetic FIXTURES ONLY, generated in pytest temporary directories.

These exercise the real orchestrator/serializer with test weather and models;
none are copied into repository artifacts or presented as production results.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from test_agent import setup_agent  # Existing isolated backend fixture.
from src.analysis_layer import forecast_facts
from src.dashboard_data import (
    artifact_fingerprint, forecast_summary, load_analysis, load_audits,
    load_forecast, load_model_metrics, load_run, matching_audit, replay_origins, run_options,
)


@pytest.fixture
def dashboard_artifacts(setup_agent, monkeypatch):
    agent, _ = setup_agent
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-a-real-key")

    def fixture_analysis(summary):
        ids = ["T1_means", "T2_peak", "limitations"]
        return dict(status="completed", selected_fact_ids=ids,
                    text="\n".join(forecast_facts(summary)[key] for key in ids),
                    openai=dict(called=True, status="completed", model="fixture-model",
                                latency_ms=1200, usage=dict(input_tokens=12, output_tokens=3, total_tokens=15)))

    agent.analysis_tool = fixture_analysis
    first = agent.run("2026-02-10", no_llm=False)
    second = agent.run("2026-02-11", no_llm=True)
    combined = pd.concat([pd.read_csv(r.path, float_precision="round_trip") for r in (first, second)], ignore_index=True)
    canonical = agent.root / "predictions/february_rolling_forecast.csv"
    combined.to_csv(canonical, index=False)
    metrics = []
    for model in ("A_direct_curve", "B_calibrated_curve", "C_hist_gradient", "D_hybrid_catboost"):
        for month in ("2025-12", "2026-01", "combined"):
            for band in ("1-24h", "25-48h", "1-48h"):
                metrics.append(dict(model_name=model, month=month, population="all_complete_observed",
                    turbine_id="ALL", lead_group=band, N=42, MAE=.123, RMSE=.234, R2=.345,
                    normalized_capacity_error_pp=12.3, bias=-.111))
    (agent.root / "metrics").mkdir()
    pd.DataFrame(metrics).to_csv(agent.root / "metrics/model_comparison.csv", index=False)
    (agent.root / "models").mkdir()
    (agent.root / "models/model_selection.json").write_text(json.dumps(dict(selected_strategy="A_direct_curve")), encoding="utf-8")
    return agent.root, first, second, combined


def test_forecast_loading_preserves_every_number_and_backend_summary(dashboard_artifacts):
    root, first, _, original = dashboard_artifacts
    frame = load_forecast(root / "predictions/february_rolling_forecast.csv").data
    assert frame is not None
    expected = original.sort_values(["forecast_origin_utc", "run_id", "turbine_id", "lead_time_hours"])
    assert np.array_equal(frame.predicted_power, expected.predicted_power)
    assert np.array_equal(frame.wind_speed_100m_ms, expected.wind_speed_100m_ms)
    one = load_run(root, first.context.forecast_origin_utc.isoformat(), first.context.run_id, frame).data
    summary = forecast_summary(one, None).data
    assert summary["turbines"] == first.summary["turbines"]
    assert summary["T1_minus_T2"] == first.summary["T1_minus_T2"]
    assert str(one.valid_time_local.dt.tz) == "Asia/Almaty"
    assert one.valid_time_local.iloc[0].hour == 1


@pytest.mark.parametrize("damage", ["out_of_bounds", "missing_lead", "future_weather", "naive_time", "mixed_run", "observations", "nonfinite_weather"])
def test_invalid_forecasts_are_rejected_without_repair(dashboard_artifacts, damage):
    root, first, _, _ = dashboard_artifacts
    frame = pd.read_csv(first.path)
    if damage == "out_of_bounds":
        frame.loc[0, "predicted_power"] = 1.01
    elif damage == "missing_lead":
        frame = frame.iloc[1:]
    elif damage == "future_weather":
        frame["weather_available_at_utc"] = "2026-02-11T00:00:00Z"
    elif damage == "naive_time":
        frame["forecast_origin_utc"] = "2026-02-09 19:00:00"
    elif damage == "mixed_run":
        frame.loc[0, "model_version"] = "another-model"
    elif damage == "observations":
        frame["actual_power"] = 0
    else:
        frame.loc[0, "wind_speed_100m_ms"] = np.inf
    frame.to_csv(first.path, index=False)
    before = first.path.read_bytes()
    result = load_forecast(first.path)
    assert result.data is None and "checks" in result.message
    assert first.path.read_bytes() == before


def test_missing_and_malformed_artifacts_are_friendly(tmp_path):
    forecast = load_forecast(tmp_path / "missing.csv")
    assert forecast.data is None
    assert forecast.command == "python -m src.cli february-forecast"
    assert load_model_metrics(tmp_path).command == "python -m src.cli phase2b-backtest"
    assert load_audits(tmp_path).data == []
    assert load_analysis(tmp_path, "missing", None, None).data is None
    folder = tmp_path / "agent_runs"
    folder.mkdir()
    (folder / "broken.json").write_text("{invalid secret payload", encoding="utf-8")
    result = load_audits(tmp_path)
    assert result.data == [] and "1 unreadable" in result.message
    assert "secret payload" not in result.message


def test_metrics_are_loaded_from_saved_artifact_not_documentation(dashboard_artifacts):
    root, *_ = dashboard_artifacts
    result = load_model_metrics(root)
    assert result.data["selected"] == "A_direct_curve"
    assert result.data["metrics"].MAE.eq(.123).all()  # Deliberately unlike real results.


def test_incomplete_metrics_are_rejected_with_friendly_message(dashboard_artifacts):
    root, *_ = dashboard_artifacts
    path = root / "metrics/model_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "N"] = np.nan
    frame.to_csv(path, index=False)
    result = load_model_metrics(root)
    assert result.data is None and "unreadable" in result.message


def test_matching_requires_run_origin_turbines_model_and_weather(dashboard_artifacts):
    root, first, second, _ = dashboard_artifacts
    audits = load_audits(root).data
    frame = load_forecast(first.path).data
    match = matching_audit(audits, frame)
    assert match["run_id"] == first.context.run_id
    assert match["analysis"]["status"] == "completed"
    assert matching_audit(audits, load_forecast(second.path).data)["analysis"]["status"] == "skipped"
    for field, wrong in (("run_id", "different"), ("forecast_origin_utc", second.context.forecast_origin_utc.isoformat()),
                         ("turbines", ["T1"]), ("model_version", "different")):
        changed = copy.deepcopy(match)
        changed[field] = wrong
        assert matching_audit([changed], frame) is None
    changed = copy.deepcopy(match)
    changed["selected_weather_runs"]["T1"]["run"] = "2026-02-09T06:00:00+00:00"
    assert matching_audit([changed], frame) is None
    origin = first.context.forecast_origin_utc.isoformat()
    assert load_run(root, origin, second.context.run_id, frame).data is None
    assert load_run(root, origin, "../../secret", frame).data is None


def test_no_secret_fields_or_arbitrary_text_returned_to_ui(dashboard_artifacts):
    root, first, _, _ = dashboard_artifacts
    path = root / "agent_runs" / f"{first.context.run_id}.json"
    raw = json.loads(path.read_text())
    secret = "SENSITIVE_FIXTURE_DO_NOT_DISPLAY"
    raw.update(api_key=secret, prompt=secret, authorization=secret, hidden_instructions=secret,
               errors=[dict(message=secret)], warnings=[secret])
    raw["tool_calls"][0]["error"] = secret
    raw["analysis"].update(text=secret, prompt=secret)
    raw["openai"].update(headers={"Authorization": secret}, input=secret)
    raw["openai"]["usage"]["request_payload"] = secret
    path.write_text(json.dumps(raw), encoding="utf-8")
    audits = load_audits(root).data
    frame = pd.read_csv(first.path)
    frame["OPENAI_API_KEY"] = secret
    frame["prompt"] = secret
    frame.to_csv(first.path, index=False)
    loaded = load_forecast(first.path).data
    audit = matching_audit(audits, loaded)
    summary = forecast_summary(loaded, audit).data
    analysis = load_analysis(root, first.context.run_id, audit, summary)
    assert secret not in json.dumps(audits)
    assert secret not in loaded.to_csv(index=False)
    assert secret not in json.dumps(analysis.data)
    assert "mean predicted normalized power" in analysis.data["text"]


def test_legacy_analysis_never_displays_unverified_free_text(dashboard_artifacts):
    root, first, _, _ = dashboard_artifacts
    path = first.path.parent / "analysis.json"
    path.write_text(json.dumps(dict(status="completed", text="unsafe arbitrary prompt")), encoding="utf-8")
    result = load_analysis(root, first.context.run_id, None, first.summary)
    assert result.data["text"] == ""
    assert "unsafe" not in str(result)
    assert "verifiable" in result.message


def test_failed_attempt_is_retained_and_never_borrows_another_forecast(dashboard_artifacts):
    root, first, _, _ = dashboard_artifacts
    raw = json.loads((root / "agent_runs" / f"{first.context.run_id}.json").read_text())
    raw.update(run_id="failed-fixture", status="failed", tool_calls=[dict(tool="validate_weather", status="failed")])
    (root / "agent_runs/failed-fixture.json").write_text(json.dumps(raw), encoding="utf-8")
    canonical = load_forecast(root / "predictions/february_rolling_forecast.csv").data
    audits = load_audits(root).data
    origin = first.context.forecast_origin_utc.isoformat()
    options = run_options(root, origin, canonical, audits)
    failed = next(option for option in options if option["run_id"] == "failed-fixture")
    assert failed["status"] == "failed" and not failed["available"]
    assert load_run(root, origin, failed["run_id"], canonical).data is None


def test_refresh_fingerprint_tracks_artifacts_and_never_env(dashboard_artifacts):
    root, first, _, _ = dashboard_artifacts
    before = artifact_fingerprint(root)
    (root / ".env").write_text("SECRET=fixture", encoding="utf-8")
    assert artifact_fingerprint(root) == before
    first.path.write_bytes(first.path.read_bytes() + b"\n")
    assert artifact_fingerprint(root) != before


def test_copied_models_use_local_files_keep_hashes_and_forecasts(setup_agent, monkeypatch):
    from src.dashboard_runtime import load_local_models, run_cached_agent
    agent, _ = setup_agent
    original = agent.run("2026-02-10", no_llm=True)
    version_dir = agent.root / "models/final" / agent.manifest["model_version"]
    version_dir.mkdir(parents=True)
    manifest = copy.deepcopy(agent.manifest)
    for turbine, model in agent.models.items():
        path = version_dir / f"{turbine}.joblib"
        joblib.dump(model, path)
        manifest["files"][turbine] = dict(path=f"Z:/original-machine/{turbine}.joblib", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path = agent.root / "models/final/manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = manifest_path.read_bytes()
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("No network allowed"))
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-a-real-key")
    new_id = run_cached_agent("2026-02-10", root=agent.root, cache_dir=agent.client.cache_dir)
    assert new_id != original.context.run_id
    assert manifest_path.read_bytes() == before
    new = pd.read_csv(agent.root / "predictions/runs" / new_id / "forecast.csv")
    old = pd.read_csv(original.path)
    pd.testing.assert_frame_equal(old.drop(columns="run_id"), new.drop(columns="run_id"))
    audit = next(a for a in load_audits(agent.root).data if a["run_id"] == new_id)
    assert audit["recalculation"] and audit["analysis"]["status"] == "skipped"
    path.write_bytes(path.read_bytes() + b"tampered-fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_local_models(agent.root)


def _app(root, monkeypatch):
    import src.dashboard as dashboard
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dashboard, "ARTIFACTS_DIR", root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dashboard.cached_catalog.clear()
    dashboard.cached_run.clear()
    def no_network(*args, **kwargs):
        pytest.fail("Dashboard rerender attempted a network request")
    monkeypatch.setattr("requests.sessions.Session.request", no_network)
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()


def test_app_missing_artifacts_starts_and_actions_are_friendly(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    assert not app.exception
    assert any("February forecast artifacts are missing" in item.value for item in app.info)
    app.selectbox[2].set_value("Run Agent Forecast").run()
    next(button for button in app.button if button.label == "Run agent forecast").click().run()
    assert not app.exception
    assert any("Frozen models are missing" in item.value for item in app.info)


def test_app_fixture_charts_values_switching_provenance_analysis_metrics(dashboard_artifacts, monkeypatch):
    root, first, second, _ = dashboard_artifacts
    app = _app(root, monkeypatch)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 3
    power_chart = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert [trace["name"] for trace in power_chart["data"]] == ["T1", "T2"]
    original = load_forecast(first.path).data
    for trace in power_chart["data"]:
        assert np.array_equal(trace["y"], original.loc[original.turbine_id.eq(trace["name"]), "predicted_power"])
    assert any("Temporal availability check passed" in item.value for item in app.success)
    assert any("mean predicted normalized power" in item.value for item in app.markdown)
    assert any(item.value == "0.1230" for item in app.metric)
    app.selectbox[1].set_value("T1").run()
    assert len(json.loads(app.get("plotly_chart")[0].proto.spec)["data"]) == 1
    next(button for button in app.button if button.label == "Next Origin →").click().run()
    assert app.selectbox[0].value == second.context.forecast_origin_utc.isoformat()
    assert not app.exception
    assert any("analysis was skipped" in item.value for item in app.info)
    assert not any("mean predicted normalized power" in item.value for item in app.markdown)


def test_app_failed_history_shows_failure_without_stale_charts(dashboard_artifacts, monkeypatch):
    root, first, _, _ = dashboard_artifacts
    raw = json.loads((root / "agent_runs" / f"{first.context.run_id}.json").read_text())
    raw.update(run_id="failed-ui-fixture", status="failed", analysis={}, openai={},
               tool_calls=[dict(tool="validate_weather", status="failed", error="SECRET_FIXTURE_BODY")])
    (root / "agent_runs/failed-ui-fixture.json").write_text(json.dumps(raw), encoding="utf-8")
    app = _app(root, monkeypatch)
    app.selectbox[3].set_value("failed-ui-fixture").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 0
    assert any("Historical execution failed at: Validate weather" in item.value for item in app.error)
    assert not any("SECRET_FIXTURE_BODY" in item.value for item in app.markdown)
    assert any("No saved forecast for this run" in item.value for item in app.info)
