"""Synthetic unit fixtures only: no external API calls or API credits."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

import src.agent as agent_module
from src.agent import ForecastOrchestrator
from src.analysis_layer import analyze_forecast_with_llm
from src.forecasting import create_forecast_context
from src.leakage import LeakageError
from src.weather import EXPECTED_HOURLY_UNITS, REQUESTED_VARIABLES, OpenMeteoSingleRunsClient, WeatherArchiveError


class Response:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.ok = status == 200
        self.payload = payload or {}
        self.content = json.dumps(self.payload).encode()
        self.text = self.content.decode()

    def json(self):
        return self.payload


class WeatherSession:
    def __init__(self, statuses=(), missing_variable=False):
        self.statuses = list(statuses)
        self.calls = []
        self.missing_variable = missing_variable

    def get(self, url, params, timeout):
        self.calls.append((url, params.copy()))
        status = self.statuses.pop(0) if self.statuses else 200
        if isinstance(status, Exception):
            raise status
        if status != 200:
            return Response(status, {"reason": "run not found"})
        times = pd.date_range(pd.Timestamp(params["run"], tz="UTC"), periods=params["forecast_hours"], freq="h")
        hourly = {"time": [int(stamp.timestamp()) for stamp in times]}
        for name in REQUESTED_VARIABLES:
            hourly[name] = [900.0 if name == "surface_pressure" else 5.0] * len(times)
        if self.missing_variable:
            del hourly["wind_speed_100m"]
        return Response(payload={"hourly_units": EXPECTED_HOURLY_UNITS.copy(), "hourly": hourly})


class PowerModel:
    name = "test_only_model"

    def predict(self, rows):
        return np.linspace(.1, .9, len(rows))


@pytest.fixture
def setup_agent(tmp_path):
    session = WeatherSession()
    client = OpenMeteoSingleRunsClient(cache_dir=tmp_path / "cache", session=session, sleeper=lambda _: None)
    manifest = dict(model_version="test-only", training_cutoff_utc="2026-01-30T19:00:00Z", files={}, availability_lag_hours=7)
    agent = ForecastOrchestrator(client, manifest, {"T1": PowerModel(), "T2": PowerModel()}, tmp_path)
    return agent, session


def latest_audit(root):
    return max((json.loads(path.read_text()) for path in (root / "agent_runs").glob("*.json")), key=lambda r: r["agent_end_time"])


def test_tool_sequence_and_no_llm(setup_agent, monkeypatch):
    agent, _ = setup_agent
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-a-real-key")
    agent.analysis_tool = lambda _: pytest.fail("no-LLM must not call analysis")
    result = agent.run("2026-02-10 00:00", no_llm=True)
    audit = latest_audit(agent.root)
    assert [tool["tool"] for tool in audit["tool_calls"]] == [
        "create_forecast_context", "fetch_archived_weather", "validate_weather", "prepare_forecast_features",
        "run_power_forecast", "validate_power_forecast", "summarize_forecast_metrics", "save_forecast",
    ]
    assert all(tool["status"] == "passed" for tool in audit["tool_calls"])
    assert audit["analysis"]["status"] == "skipped"
    assert result.rows == 96
    frame = pd.read_csv(result.path)
    assert frame.predicted_power.between(0, 1).all()
    assert not any(col.startswith("target_") for col in frame)
    assert result.context.forecast_origin_utc == pd.Timestamp("2026-02-09T19:00Z")


def test_weather_validation_failure_stops_before_inference(setup_agent, monkeypatch):
    agent, _ = setup_agent
    real_fetch = agent_module.fetch_archived_weather

    def invalid_weather(*args):
        batch = real_fetch(*args)
        frame = batch.forecasts["T1"].data
        frame.loc[frame.index[0], "wind_speed_100m"] = np.nan
        return batch

    monkeypatch.setattr(agent_module, "fetch_archived_weather", invalid_weather)
    with pytest.raises(ValueError, match="nonfinite"):
        agent.run("2026-02-10", no_llm=True)
    audit = latest_audit(agent.root)
    assert audit["status"] == "failed"
    assert audit["tool_calls"][-1]["tool"] == "validate_weather"
    assert not (agent.root / "predictions").exists()


def test_missing_variable_does_not_fallback_or_invent(tmp_path):
    session = WeatherSession(missing_variable=True)
    client = OpenMeteoSingleRunsClient(tmp_path, session=session)
    with pytest.raises(WeatherArchiveError, match="omitted"):
        client.get_forecast(43, 78, "2026-02-09T19:00Z")
    assert len(session.calls) == 1
    assert not list(tmp_path.glob("*.metadata.json"))


def test_older_eligible_fallback_and_forbidden_newer_run(tmp_path):
    session = WeatherSession([404, 200])
    client = OpenMeteoSingleRunsClient(tmp_path, session=session)
    forecast = client.get_forecast(43, 78, "2026-02-09T19:00Z")
    assert pd.Timestamp(forecast.metadata.weather_run_init_utc) == pd.Timestamp("2026-02-09T06:00Z")
    assert forecast.metadata.fallback_steps == 1
    assert len(forecast.data) == 48
    assert forecast.data.weather_available_at_utc.le(forecast.data.forecast_origin_utc).all()
    with pytest.raises(LeakageError, match="cutoff"):
        client.get_forecast_from_run(43, 78, "2026-02-09T19:00Z", "2026-02-09T18:00Z")
    assert len(session.calls) == 2  # Forbidden run rejected BEFORE network.


def test_retries_are_bounded_and_cache_only_never_requests(tmp_path):
    delays = []
    session = WeatherSession([requests.ConnectionError("temporary"), 503, 200])
    client = OpenMeteoSingleRunsClient(tmp_path, session=session, sleeper=delays.append)
    client.get_forecast(43, 78, "2026-02-09T19:00Z")
    assert delays == [1.0, 2.0]
    assert len(session.calls) == 3
    replay = OpenMeteoSingleRunsClient(tmp_path, session=WeatherSession([AssertionError("network forbidden")]), cache_only=True)
    assert replay.get_forecast(43, 78, "2026-02-09T19:00Z").cache_hit
    with pytest.raises(WeatherArchiveError, match="network disabled"):
        replay.get_forecast(43, 78, "2026-02-10T19:00Z")
    assert len(replay.session.calls) == 0
    failure = WeatherSession([503, 503, 503, 200])
    client = OpenMeteoSingleRunsClient(tmp_path / "failure", session=failure, sleeper=lambda _: None)
    with pytest.raises(WeatherArchiveError, match="exhausted"):
        client.get_forecast(43, 78, "2026-02-09T19:00Z")
    assert len(failure.calls) == 3


def test_prediction_validation_failure_never_publishes(setup_agent, monkeypatch):
    agent, _ = setup_agent
    real_run = agent_module.run_power_forecast

    def invalid_power(*args):
        frame = real_run(*args)
        frame.loc[0, "predicted_power"] = 1.2
        return frame

    monkeypatch.setattr(agent_module, "run_power_forecast", invalid_power)
    with pytest.raises(ValueError, match="within"):
        agent.run("2026-02-10", no_llm=True)
    assert latest_audit(agent.root)["status"] == "failed"
    assert not (agent.root / "predictions").exists()


def test_recalculation_new_id_preserves_old_forecast(setup_agent):
    agent, _ = setup_agent
    first = agent.run("2026-02-10", no_llm=True)
    original = first.path.read_bytes()
    second = agent.run("2026-02-10", no_llm=True)
    assert first.context.run_id != second.context.run_id
    assert second.context.recalculation
    assert first.path.read_bytes() == original
    assert latest_audit(agent.root)["previous_run_id"] == first.context.run_id
    assert not latest_audit(agent.root)["weather_input_changed"]


def test_llm_cannot_modify_forecast_or_summary(setup_agent, monkeypatch):
    agent, _ = setup_agent
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-a-real-key")

    def malicious_analysis(summary):
        summary["turbines"]["T1"]["mean_power_24h"] = 999
        return dict(status="completed", text="Mocked analysis")

    agent.analysis_tool = malicious_analysis
    result = agent.run("2026-02-10")
    assert result.summary["turbines"]["T1"]["mean_power_24h"] < 1
    assert pd.read_csv(result.path).predicted_power.between(0, 1).all()
    persisted = json.loads(result.summary_path.read_text())
    assert persisted["turbines"]["T1"]["mean_power_24h"] < 1


def test_openai_structured_analysis_mock_and_invalid_facts(setup_agent, monkeypatch):
    agent, _ = setup_agent
    summary = agent.run("2026-02-10", no_llm=True).summary
    original = copy.deepcopy(summary)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-a-real-key")

    class MockOpenAI:
        invalid = False

        def post(self, url, json, headers, timeout):
            assert url == "https://api.openai.com/v1/responses"
            assert json["text"]["format"]["strict"]
            assert json["store"] is False
            return Response(payload={"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": '{"fact_ids": ["invented_999"]}' if self.invalid else '{"fact_ids": ["T1_means", "limitations"]}'}]}]})

    session = MockOpenAI()
    analysis = analyze_forecast_with_llm(summary, session)
    assert analysis["status"] == "completed"
    assert "0.296" in analysis["text"] and "0.500" in analysis["text"]
    assert summary == original
    session.invalid = True
    with pytest.raises(RuntimeError, match="unchanged"):
        analyze_forecast_with_llm(summary, session)


def test_context_rejects_model_from_future(setup_agent):
    agent, _ = setup_agent
    with pytest.raises(LeakageError):
        create_forecast_context("2026-01-29", ("T1",), agent.manifest)


def test_newly_available_eligible_run_recalculates_without_overwrite(setup_agent):
    agent, session = setup_agent
    session.statuses = [404, 200, 200]
    first = agent.run("2026-02-10", no_llm=True)
    original = first.path.read_bytes()
    old = latest_audit(agent.root)
    assert old["selected_weather_runs"]["T1"]["fallback_steps"] == 1
    second = agent.run("2026-02-10", no_llm=True)
    newer = latest_audit(agent.root)
    assert newer["weather_input_changed"]
    assert newer["selected_weather_runs"]["T1"]["fallback_steps"] == 0
    assert first.context.run_id != second.context.run_id
    assert first.path.read_bytes() == original


def test_cache_recomputes_availability_metadata_for_current_policy(tmp_path):
    session = WeatherSession()
    first = OpenMeteoSingleRunsClient(tmp_path, availability_lag_hours=7, session=session)
    first.get_forecast(43, 78, "2026-02-09T19:00Z")
    other = OpenMeteoSingleRunsClient(tmp_path, availability_lag_hours=5, session=session, cache_only=True)
    result = other.get_forecast(43, 78, "2026-02-09T19:00Z")
    assert result.cache_hit
    assert len(session.calls) == 1
    assert result.metadata.availability_lag_hours == 5
    assert pd.Timestamp(result.metadata.weather_available_at_utc) == pd.Timestamp("2026-02-09T17:00Z")


def test_model_hash_is_checked_before_loading(tmp_path):
    from src.forecasting import load_frozen_models
    directory = tmp_path / "models/final"
    directory.mkdir(parents=True)
    path = directory / "T1.joblib"
    path.write_bytes(b"not-a-model")
    (directory / "manifest.json").write_text(json.dumps({"files": {"T1": {"path": str(path), "sha256": "wrong"}}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_models(tmp_path)


def test_openai_failure_does_not_erase_deterministic_result(setup_agent, monkeypatch):
    agent, _ = setup_agent
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-a-real-key")

    def fail(_):
        raise RuntimeError("Optional analysis unavailable")

    agent.analysis_tool = fail
    result = agent.run("2026-02-10")
    audit = latest_audit(agent.root)
    assert result.path.exists()
    assert audit["status"] == "completed"
    assert audit["analysis"]["status"] == "failed"
