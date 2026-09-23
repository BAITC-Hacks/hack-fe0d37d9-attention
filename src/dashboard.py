"""Streamlit presentation layer over immutable forecasting artifacts."""
from __future__ import annotations

import os
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import ARTIFACTS_DIR, TURBINES
from .dashboard_data import (
    DISPLAY_TZ, MODEL_LABELS, TOOL_LABELS, Loaded, artifact_fingerprint,
    forecast_summary, load_analysis, load_audits, load_forecast, load_model_metrics,
    load_run, local_time, matching_audit, replay_origins, run_options,
)

COLORS = {"T1": "#087f8c", "T2": "#c46a21"}
STATUS_LABELS = {"passed": "✓ Passed", "completed": "✓ Completed", "failed": "✕ Failed",
                 "running": "◌ Running", "skipped": "— Skipped", "not recorded": "— Not recorded"}
CSS = """
<style>
.block-container {max-width: 1440px; padding-top: 4rem; padding-bottom: 3rem;}
h1 {letter-spacing: -.055em; font-weight: 800 !important;}
h2,h3 {letter-spacing: -.025em;}
[data-testid="stMetricValue"] {font-size: 1.65rem; font-weight: 650;}
[data-testid="stMetricLabel"] {color: #5a6d79;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
.eyebrow {color:#087f8c; font-size:.75rem; font-weight:750; letter-spacing:.15em; margin-bottom:6px;}
.subtitle {font-size:1.22rem; color:#47606f; margin-top:-14px;}
.pipeline {display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; margin:12px 0 18px;}
.step {background:#fff; border:1px solid #dce5e9; border-radius:10px; padding:15px; min-height:112px;}
.step-number {color:#738793; font-size:.72rem; letter-spacing:.08em; margin-bottom:7px;}
.step-title {font-size:.92rem; font-weight:650; line-height:1.4; margin-bottom:8px;}
.badge {font-size:.76rem; color:#266575; background:#e8f4f4; border-radius:5px; padding:3px 7px; display:inline-block;}
.failed {color:#a32424; background:#fcecec;}
.neutral {color:#526570; background:#eef1f4;}
.architecture {display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:18px 0;}
.node {border:1px solid #d5e2e6; background:white; border-radius:9px; padding:14px; font-size:.88rem;}
.guard {border-left:4px solid #087f8c; background:#eaf5f5; border-radius:6px; padding:14px;}
.kpi-strip {display:grid; grid-template-columns:.5fr 1fr 1fr 1fr 1.35fr; gap:12px; align-items:center; background:white; border:1px solid #dce5e9; border-radius:12px; padding:16px 20px;}
.kpi-label {font-size:.75rem; color:#5a6d79; margin-bottom:4px;}
.kpi-value {font-size:1.45rem; font-weight:700; color:#172d3b; line-height:1.3;}
@media(max-width:700px) {.block-container {padding:4rem 1rem 1rem;} .pipeline {grid-template-columns:1fr;} .kpi-strip {grid-template-columns:1fr 1fr;} .kpi-strip>div:first-child {grid-column:1/-1;}}
</style>
"""


@st.cache_data(show_spinner=False, max_entries=8)
def cached_catalog(root: str, fingerprint: tuple):
    folder = Path(root)
    return (load_forecast(folder / "predictions/february_rolling_forecast.csv"),
            load_audits(folder), load_model_metrics(folder))


@st.cache_data(show_spinner=False, max_entries=64)
def cached_run(root: str, fingerprint: tuple, origin: str, run_id: str):
    canonical, audits, _ = cached_catalog(root, fingerprint)
    result = load_run(Path(root), origin, run_id, canonical.data)
    audit = matching_audit(audits.data, result.data) if result.data is not None else next(
        (a for a in audits.data if a["run_id"] == run_id and a["forecast_origin_utc"] == origin), None)
    summary = forecast_summary(result.data, audit) if result.data is not None else Loaded()
    analysis = load_analysis(Path(root), run_id, audit, summary.data)
    return result, audit, summary, analysis


def notice(result: Loaded):
    if result.message:
        st.info(result.message)
    if result.command:
        st.code(result.command, language="bash")


def section_label(number: str, text: str):
    st.markdown(f'<div class="eyebrow">{escape(number)} / {escape(text)}</div>', unsafe_allow_html=True)


def time_chart(frame: pd.DataFrame, value: str, title: str, height: int = 290) -> go.Figure:
    fig = go.Figure()
    for turbine, group in frame.groupby("turbine_id"):
        group = group.sort_values("valid_time_utc")
        # Axis is explicitly local wall time, preventing the browser's timezone from
        # changing labels. The UTC source columns remain unchanged.
        times = group.valid_time_local.dt.tz_localize(None)
        fig.add_trace(go.Scatter(x=times.tolist(), y=group[value].tolist(), name=turbine,
            mode="lines", line=dict(color=COLORS[turbine], width=3, dash="solid" if turbine == "T1" else "dash"),
            customdata=group.valid_time_local.dt.strftime("%Y-%m-%d %H:%M Asia/Almaty").tolist(),
            hovertemplate="%{customdata}<br>" + title + ": %{y:.3f}<extra>" + turbine + "</extra>"))
    origin = frame.forecast_origin_local.iloc[0].tz_localize(None)
    fig.add_shape(type="line", x0=origin, x1=origin, y0=0, y1=1, yref="paper",
                  line=dict(color="#8c9ca5", width=1, dash="dot"))
    fig.update_layout(height=height, margin=dict(l=12, r=16, t=25, b=15), template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#47606f"),
        hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(title="Valid time · Asia/Almaty (UTC+05)", tickformat="%b %d\n%H:%M",
                   range=[origin, frame.valid_time_local.max().tz_localize(None)], showgrid=False),
        yaxis=dict(title=title, gridcolor="#e2e9ed", zeroline=False))
    if value == "predicted_power":
        fig.update_yaxes(range=[0, 1], dtick=.2)
        fig.add_annotation(x=origin, y=1, yref="paper", text="Origin", showarrow=False, xanchor="left")
    return fig


def render_kpis(summary: dict | None, turbines: list[str]):
    if not summary:
        return
    for turbine in turbines:
        values = summary["turbines"].get(turbine)
        if not values:
            continue
        items = [("Next 24h average", f"{values['mean_power_24h']:.3f}"),
                 ("Next 48h average", f"{values['mean_power_48h']:.3f}"),
                 ("Peak predicted power", f"{values['peak_power']:.3f}"),
                 ("Peak local time", local_time(values["peak_time_local"], True))]
        cells = f'<div class="kpi-value" style="color:{COLORS[turbine]}">{escape(turbine)}</div>'
        cells += "".join(f'<div><div class="kpi-label">{escape(label)}</div><div class="kpi-value">{escape(value)}</div></div>' for label, value in items)
        st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)


def render_provenance(frame: pd.DataFrame):
    section_label("03", "POINT-IN-TIME PROVENANCE")
    st.subheader("Why this forecast is leakage-safe")
    st.success("✓ Temporal availability check passed · verified by the existing backend validator")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Forecast origin:** {local_time(frame.forecast_origin_utc.iloc[0])} Asia/Almaty")
        st.markdown(f"**Forecast origin UTC:** {frame.forecast_origin_utc.iloc[0].strftime('%Y-%m-%d %H:%M UTC')}")
    with right:
        st.code("weather_available_at <= forecast_origin", language="text")
    rows = []
    for turbine, group in frame.groupby("turbine_id"):
        row = group.iloc[0]
        rows.append({"Turbine": turbine, "ECMWF initialization (UTC)": row.weather_run_init_utc.strftime("%Y-%m-%d %H:%M"),
                     "Weather available (UTC)": row.weather_available_at_utc.strftime("%Y-%m-%d %H:%M"),
                     "Availability lag (h)": row.availability_lag_hours,
                     "Run age at origin (h)": (row.forecast_origin_utc - row.weather_run_init_utc).total_seconds() / 3600})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.write("The system uses only the archived weather forecast that could have been available at the forecast origin. Actual future weather is never used as a predictor.")
    st.caption("Availability is based on the recorded conservative publication lag, not measured publication telemetry. Internal clocks remain UTC.")


def render_insights(summary: dict | None, turbines: list[str]):
    with st.expander("Forecast insights · deterministic backend metrics", expanded=True):
        if not summary:
            st.info("Deterministic insights are unavailable for this artifact.")
            return
        rows = []
        for turbine in turbines:
            if turbine not in summary["turbines"]:
                continue
            values = summary["turbines"][turbine]
            low = values["lowest_3h_period"]
            rows.extend([
                {"Turbine": turbine, "Insight": "Peak generation", "Normalized power / change": f"{values['peak_power']:.3f}", "Local period": local_time(values["peak_time_local"], True)},
                {"Turbine": turbine, "Insight": "Lowest hour", "Normalized power / change": f"{values['minimum_power']:.3f}", "Local period": local_time(values["minimum_time_local"], True)},
                {"Turbine": turbine, "Insight": "Lowest 3h block", "Normalized power / change": f"{low['mean_power']:.3f}", "Local period": f"{local_time(low['start_local'], True)} → {local_time(low['end_local'], True)}"},
            ])
            for ramp in values["ramps"]:
                rows.append({"Turbine": turbine, "Insight": "Strongest ramp-up" if ramp["kind"] == "ramp_up" else "Strongest ramp-down",
                             "Normalized power / change": f"{ramp['change_normalized_power']:+.3f}",
                             "Local period": f"{local_time(ramp['from_local'], True)} → {local_time(ramp['to_local'], True)}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        difference = summary.get("T1_minus_T2")
        if difference and len(turbines) == 2:
            st.write(f"**T1 − T2 mean power:** next 24h **{difference['mean_power_24h']:+.3f}** · next 48h **{difference['mean_power_48h']:+.3f}**")
        st.caption("Summaries use the existing backend's deterministic summarizer on saved predictions. Ramp values are signed one-hour changes; ties follow backend ordering. All times are Asia/Almaty.")


def render_execution(audit: dict | None, origin: str):
    section_label("04", "DETERMINISTIC ORCHESTRATION")
    st.subheader("Agent Execution")
    if audit is None:
        st.info("No saved agent run for this origin and forecast run ID.")
        st.caption("A persisted forecast alone does not prove that the orchestrator executed. No tool statuses are inferred.")
        st.code(f'python -m src.cli agent-forecast --origin "{local_time(origin)}" --cache-only --no-llm', language="bash")
        return
    status = STATUS_LABELS.get(audit["status"], audit["status"])
    st.write(f"**Run status: {status}** · `{audit['run_id']}`")
    st.caption("Recorded execution covers " + " / ".join(audit["turbines"]) + "; the turbine chart filter does not change this audit.")
    steps = []
    for index, step in enumerate(audit["tool_calls"], 1):
        label = TOOL_LABELS.get(step["tool"], "Unrecognized recorded tool")
        status = STATUS_LABELS.get(step["status"], step["status"])
        if step["tool"] == "fetch_archived_weather" and step["status"] == "passed":
            runs = audit["selected_weather_runs"]
            if set(runs) == set(audit["turbines"]) and all(run["cache_hit"] is True for run in runs.values()):
                status += " · ● Cache"
        if step["tool"] == "analyze_forecast_with_llm":
            status += " · " + STATUS_LABELS.get(audit["analysis"]["status"], audit["analysis"]["status"])
        duration = f" · {step['duration_s']:.2f}s" if step["duration_s"] is not None else ""
        kind = "failed" if step["status"] == "failed" else ""
        steps.append(f'<div class="step"><div class="step-number">{index:02d}{escape(duration)}</div><div class="step-title">{escape(label)}</div><span class="badge {kind}">{escape(status)}</span></div>')
    st.markdown('<div class="pipeline">' + "".join(steps) + "</div>", unsafe_allow_html=True)
    if not any(step["tool"] == "analyze_forecast_with_llm" for step in audit["tool_calls"]):
        st.caption("OpenAI analysis: " + STATUS_LABELS.get(audit["analysis"]["status"], audit["analysis"]["status"]) + " · no analysis tool call recorded")
    if audit["status"] == "failed" or audit["error_count"]:
        failed = [TOOL_LABELS.get(s["tool"], "Recorded tool") for s in audit["tool_calls"] if s["status"] == "failed"]
        st.error("Historical execution failed" + (" at: " + ", ".join(failed) if failed else "; see the local audit for the failure location") + ". Later steps are not reported as passed.")
    if audit["warning_count"]:
        st.warning(f"⚠ {audit['warning_count']} warning(s) recorded. Known modelling limitations remain applicable; raw diagnostic payloads are not displayed.")
    for turbine, run in audit["selected_weather_runs"].items():
        if run["fallback_steps"]:
            st.warning(f"{turbine}: an older eligible ECMWF run was used ({int(run['fallback_steps'])} initialization steps).")
    if audit["recalculation"]:
        st.caption(f"Recalculation of run {audit['previous_run_id'] or 'not recorded'} · weather inputs changed: {audit['weather_input_changed']} · prior outputs preserved")


def execute_agent(origin: str, with_ai: bool = False):
    """Explicit UI action delegates the entire write/audit lifecycle to the backend."""
    if not (ARTIFACTS_DIR / "models/final/manifest.json").is_file():
        st.info("Frozen models are missing. Restore artifacts/models/final/ or run the validated reproduction workflow first.")
        st.code("python -m src.cli freeze-models", language="bash")
        return
    if with_ai and not os.getenv("OPENAI_API_KEY"):
        st.info("AI analysis is not configured. Existing saved analysis remains viewable without an API key.")
        return
    try:
        from .dashboard_runtime import run_cached_agent
        with st.spinner("Executing the existing agent with cached weather…"):
            run_id = run_cached_agent(origin, with_ai=with_ai, root=ARTIFACTS_DIR)
        st.session_state["new_run"] = run_id
        st.session_state["run_feedback"] = "Saved a new audited run. Previous outputs remain preserved."
        cached_catalog.clear()
        cached_run.clear()
        st.rerun()
    except Exception:
        # Provider/OS exception text can contain paths, headers, or credentials.
        # The backend records the actual failure; only a fixed message reaches UI.
        cached_catalog.clear()
        cached_run.clear()
        st.error("The agent could not complete this run. Check the frozen model files and eligible weather cache. No successful status has been invented; any saved failure appears in the run selector after refresh.")
        st.code(f'python -m src.cli agent-forecast --origin "{local_time(origin)}" --cache-only --no-llm', language="bash")


def render_analysis(result: Loaded, origin: str):
    section_label("05", "AI-ASSISTED ANALYSIS")
    st.subheader("What OpenAI concluded")
    st.caption("AI explanation does not modify numerical forecasts.")
    analysis = result.data
    if analysis:
        cols = st.columns(4)
        cols[0].metric("GPT model", analysis["model"] or "Not recorded")
        cols[1].metric("Status", analysis["status"].capitalize())
        cols[2].metric("Latency", f"{analysis['latency_ms']/1000:.2f} s" if analysis["latency_ms"] is not None else "Not recorded")
        cols[3].metric("Total tokens", str(analysis["usage"]["total_tokens"]) if analysis["usage"]["total_tokens"] is not None else "Not recorded")
        usage = analysis["usage"]
        st.caption(f"Input tokens: {usage['input_tokens'] if usage['input_tokens'] is not None else 'not recorded'} · Output tokens: {usage['output_tokens'] if usage['output_tokens'] is not None else 'not recorded'}")
        if analysis["text"]:
            with st.container(border=True):
                st.write(analysis["text"])
        elif analysis["status"] == "failed":
            st.warning("Optional AI analysis failed. The persisted numerical forecast remains available.")
        elif analysis["status"] == "skipped":
            st.info("AI analysis was skipped for this run. Deterministic forecasts and insights are available without it.")
    notice(result)
    with st.expander("Generate an explanation explicitly"):
        st.write("This button runs the existing agent again for both turbines using cached weather, then requests OpenAI analysis. It saves a new forecast, analysis, and audit together and selects that run. Provider usage may incur a charge.")
        st.caption("Changing dates, turbines, or tabs never triggers an API call. The OpenAI key is read only by the existing backend.")
        if st.button("Run AI analysis", disabled=not bool(os.getenv("OPENAI_API_KEY"))):
            execute_agent(origin, with_ai=True)
        if not os.getenv("OPENAI_API_KEY"):
            st.caption("No API key configured. Viewing persisted analysis requires no key.")


def render_validation(result: Loaded):
    with st.expander("Model Validation · held-out December / January", expanded=False):
        if result.data is None:
            notice(result)
            st.caption("Restore the saved comparison to view results. Full backtesting is an offline setup step and overwrites comparison artifacts.")
            return
        frame, selected = result.data["metrics"], result.data["selected"]
        primary = frame.loc[frame.population.eq("all_complete_observed") & frame.turbine_id.eq("ALL")]
        comparison = primary.loc[primary.lead_group.eq("1-48h")].pivot(index="model_name", columns="month", values="MAE")
        comparison = comparison.reindex(list(MODEL_LABELS)).rename(columns={"2025-12": "December MAE", "2026-01": "January MAE", "combined": "Combined MAE"})
        comparison.index = [MODEL_LABELS[name] + (" · SELECTED" if name == selected else "") for name in comparison.index]
        st.dataframe(comparison.style.format("{:.4f}", na_rep="Not recorded"), width="stretch")
        if selected:
            st.write(f"**Selected: {MODEL_LABELS[selected]}** · saved model-selection artifact")
            chosen = primary.loc[primary.model_name.eq(selected) & primary.month.eq("combined")]
            for column, band in zip(st.columns(3), ("1-48h", "1-24h", "25-48h")):
                rows = chosen.loc[chosen.lead_group.eq(band)]
                column.metric(f"MAE · {band}", f"{rows.iloc[0].MAE:.4f}" if not rows.empty else "Not recorded")
            total = chosen.loc[chosen.lead_group.eq("1-48h")]
            if not total.empty:
                st.write(f"Residual bias: **{total.iloc[0].bias:+.4f}** · RMSE: **{total.iloc[0].RMSE:.4f}** · scored pairs: **{int(total.iloc[0].N):,}**")
        else:
            st.info("The model-selection artifact is missing or unsupported; no winning model is inferred.")
        st.warning("Known limitations: high generation is underpredicted; longer horizons were less accurate in the held-out validation. No calibrated probabilistic uncertainty interval is available.")
        st.caption("Primary population: all complete observed hours, including suspected unavailability. Validation month refers to the local forecast origin. Overlapping forecast/target pairs are not independent unique hours. These are pre-February results, not February accuracy.")


def render_replay(root: Path, canonical, audits, origin: str, run_id: str | None, frame, analysis):
    st.subheader("February Replay")
    st.write("A daily cycle: new historical origin → eligible ECMWF run → 48-hour forecast → optional analysis → saved run.")
    st.caption("Forecast persistence occurs before optional AI analysis. Use Previous / Next Origin in the top bar to replay saved days. No calls run automatically.")
    if frame is not None:
        rows = frame[["turbine_id", "weather_run_init_utc"]].drop_duplicates()
        weather = " · ".join(f"{row.turbine_id}: {row.weather_run_init_utc.strftime('%b %d %H:%M UTC')}" for row in rows.itertuples())
        st.success(f"{local_time(origin)} local → {weather} → {len(frame)} saved hourly predictions → AI: {analysis.data['status'] if analysis.data else 'not recorded'} → run {run_id}")
    timeline = []
    for day in replay_origins():
        options = run_options(root, day, canonical, audits)
        day_audits = [audit for audit in audits if audit["forecast_origin_utc"] == day]
        completed = sum(audit["status"] == "completed" for audit in day_audits)
        failed = sum(audit["status"] == "failed" for audit in day_audits)
        latest = max(day_audits, key=lambda audit: audit["agent_end_time"] or "") if day_audits else None
        runs = options[0]["run_id"] if options else None
        timeline.append({"Origin · Asia/Almaty": local_time(day), "Forecast": "● Saved" if any(o["available"] for o in options) else "Missing",
                         "Latest agent status": STATUS_LABELS.get(latest["status"], latest["status"]) if latest else "No saved agent run",
                         "Completed attempts": completed, "Failed attempts": failed,
                         "Saved AI explanations": sum(a["analysis"]["status"] == "completed" for a in day_audits),
                         "Preferred run": runs or "—"})
    st.dataframe(pd.DataFrame(timeline), hide_index=True, width="stretch", height=520)
    st.caption("A saved forecast is distinct from a completed agent execution. Failed historical attempts remain counted even when a later retry succeeded.")
    with st.expander("Recalculate the complete February cycle offline"):
        st.write("With frozen models and all archived weather responses in place, the existing replay command repeats all 29 origins, saves new audits, and verifies exact equality with canonical predictions. OpenAI is disabled.")
        st.code("python -m src.cli replay-february", language="bash")


def render_system(frame, metrics: Loaded):
    st.subheader("System architecture")
    labels = ["Historical Forecast Origin", "ECMWF Single Run", "Weather Validation", "Feature Engineering", "Power Forecast Model", "Forecast Validation", "Persistence", "OpenAI Analysis", "Dashboard"]
    st.markdown('<div class="guard"><b>TemporalLeakageGuard</b> · deterministic checks around forecast context, archived-weather availability, and training cutoffs.</div>', unsafe_allow_html=True)
    st.markdown('<div class="architecture">' + '<span aria-hidden="true">→</span>'.join(f'<div class="node">{label}</div>' for label in labels) + '</div>', unsafe_allow_html=True)
    st.write("One deterministic orchestrator executes the tools and records the result. OpenAI selects salient facts after the forecast has been validated and saved. The dashboard reads those outputs.")
    with st.expander("Technical Details", expanded=True):
        st.dataframe(pd.DataFrame([{"Turbine": t.turbine_id, "Latitude": t.latitude, "Longitude": t.longitude} for t in TURBINES.values()]), hide_index=True, width="stretch")
        st.markdown("**Primary wind proxy:** 100 m · **Forecast horizon:** 48h  \n**Internal timezone:** UTC · **Display timezone:** Asia/Almaty  \n**Weather provider:** Open-Meteo Single Runs · **Weather model:** ECMWF IFS (`ecmwf_ifs`)")
        model = frame.model_name.iloc[0] if frame is not None else (metrics.data["selected"] if metrics.data else None)
        lag = ", ".join(f"{value:g}h" for value in frame.availability_lag_hours.unique()) if frame is not None else "Not recorded in this checkout"
        st.markdown(f"**Selected forecasting model:** {MODEL_LABELS.get(model, model or 'Not recorded in this checkout')}  \n**Recorded availability lag:** {lag}")
        st.caption("The documented baseline uses separate empirical power curves and a conservative 7h lag. Actual model identity and lag above come from saved artifacts.")
        st.markdown("**Assumptions and limitations**\n\n- SCADA timezone convention is not confirmed by the organizer.\n- Actual turbine hub height is unknown; 100 m is a proxy.\n- The 7h publication lag is a conservative assumption.\n- No calibrated probabilistic uncertainty interval is available.\n- The selected model underpredicts some high-output periods.\n- Validation covers two winter months; no February ground-truth accuracy is claimed.")


def main():
    st.set_page_config(page_title="WindAgent AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### WindAgent AI")
        st.caption("HACKALEM · ENERGY INTELLIGENCE")
        st.divider()
        st.markdown("**Your two-minute demo**\n\n1. Select a February forecast date.\n2. Inspect T1 / T2 generation.\n3. Open Agent Execution.\n4. Inspect leakage-safe provenance.\n5. Review the saved AI analysis.\n6. Switch dates to show the repeating cycle.")
        st.divider()
        st.caption("Cached Replay reads local artifacts. No internet or API key is required to view saved forecasts and explanations.")
        if st.button("Refresh saved artifacts", width="stretch"):
            cached_catalog.clear()
            cached_run.clear()
            st.rerun()
    section_label("HACKALEM", "WIND POWER OPERATIONS")
    st.title("WindAgent AI")
    st.markdown('<div class="subtitle">Agentic 48-hour Wind Power Forecasting</div>', unsafe_allow_html=True)
    st.caption("Point-in-time ECMWF weather forecasts → deterministic forecasting engine → autonomous agent → AI-assisted analysis")
    fingerprint = artifact_fingerprint(ARTIFACTS_DIR)
    forecast, audit_result, metrics = cached_catalog(str(ARTIFACTS_DIR), fingerprint)
    canonical, audits = forecast.data, audit_result.data
    origins = sorted(set(a["forecast_origin_utc"] for a in audits) | (set(canonical.forecast_origin_utc.map(pd.Timestamp.isoformat)) if canonical is not None else set()))
    origins = origins or replay_origins()
    default = next((i for i, day in enumerate(origins) if local_time(day).startswith("2026-02-10")), 0)
    if st.session_state.get("origin") not in origins:
        st.session_state["origin"] = origins[default]
    def move_origin(offset):
        pos = origins.index(st.session_state["origin"])
        st.session_state["origin"] = origins[max(0, min(len(origins)-1, pos + offset))]
    with st.container(border=True):
        control = st.columns([2.2, 1, 1.3])
        origin = control[0].selectbox("Forecast origin · Asia/Almaty", origins, key="origin", format_func=local_time)
        turbine_filter = control[1].selectbox("Turbines", ["Both", "T1", "T2"])
        mode = control[2].selectbox("Mode", ["Cached Replay", "Run Agent Forecast"])
        nav = st.columns([1, 1, 3])
        nav[0].button("← Previous Origin", on_click=move_origin, args=(-1,), disabled=origin == origins[0], width="stretch")
        nav[1].button("Next Origin →", on_click=move_origin, args=(1,), disabled=origin == origins[-1], width="stretch")
        nav[2].caption("● LOCAL ARTIFACTS · 48 HOURLY LEADS · POWER NORMALIZED 0–1")
        if mode == "Run Agent Forecast":
            st.caption("Explicitly execute the existing agent for both turbines. Weather is cache-only; OpenAI is disabled. Missing cache fails cleanly. A new immutable run is saved.")
            if st.button("Run agent forecast", type="primary"):
                execute_agent(origin)
    if feedback := st.session_state.pop("run_feedback", None):
        st.success(feedback)
    if audit_result.message:
        st.warning(audit_result.message)
    if canonical is None:
        notice(forecast)
        with st.expander("Prepare this checkout for the demo", expanded=not audits):
            st.write("Copy the existing Phase 4 artifacts into this repository's artifacts/ folder. Viewing saved forecasts does not require the weather cache or model binaries. To generate artifacts from scratch, follow the complete README reproduction workflow, including model validation and freezing.")
            st.code("artifacts/predictions/february_rolling_forecast.csv\nartifacts/predictions/runs/\nartifacts/agent_runs/\nartifacts/metrics/model_comparison.csv\nartifacts/models/model_selection.json", language="text")
    options = run_options(ARTIFACTS_DIR, origin, canonical, audits)
    frame = audit = summary = run_id = None
    analysis = Loaded(message="No saved AI analysis for this run.")
    if options:
        ids = [option["run_id"] for option in options]
        labels = {option["run_id"]: f"{option['run_id'][:12]} · {option['status']} · AI {option['analysis']}" for option in options}
        run_key = "run_" + origin
        new_run = st.session_state.pop("new_run", None)
        if new_run in ids:
            st.session_state[run_key] = new_run
        elif st.session_state.get(run_key) not in ids:
            st.session_state[run_key] = ids[0]
        run_id = st.selectbox("Saved run · exact forecast / audit match", ids, format_func=labels.get, key=run_key)
        result, audit, summary_result, analysis = cached_run(str(ARTIFACTS_DIR), fingerprint, origin, run_id)
        frame, summary = result.data, summary_result.data
        if frame is None:
            notice(result)
        else:
            notice(summary_result)
            if audit is None and any(a["run_id"] == run_id for a in audits):
                st.warning("The audit metadata does not match this saved forecast. Its execution and analysis are not attributed to these predictions.")
                analysis = Loaded(message="Analysis hidden because the audit does not match the forecast.")
    turbines = ["T1", "T2"] if turbine_filter == "Both" else [turbine_filter]
    if frame is not None:
        st.caption(f"FORECAST HORIZON: 48h · WEATHER: ECMWF IFS · MODEL: {MODEL_LABELS.get(frame.model_name.iloc[0], frame.model_name.iloc[0])} · VERSION: {frame.model_version.iloc[0]}")
        runs = frame[["turbine_id", "weather_run_init_utc"]].drop_duplicates()
        st.caption("SELECTED ECMWF RUN (UTC): " + " · ".join(f"{r.turbine_id} {r.weather_run_init_utc:%Y-%m-%d %H:%M}" for r in runs.itertuples()))
        render_kpis(summary, turbines)
    forecast_tab, agent_tab, replay_tab, system_tab = st.tabs(["Generation & weather", "Agent Execution & AI", "February Replay", "System & validation"])
    with forecast_tab:
        if frame is not None:
            visible = frame.loc[frame.turbine_id.isin(turbines)]
            if visible.empty:
                st.info("This saved run does not contain the selected turbine. Select another turbine or run.")
            else:
                section_label("01", "GENERATION OUTLOOK")
                st.subheader("Next 48 hours")
                st.caption("Normalized power: 0 = no output · 1 = rated output. Each line is one turbine's forecast from the selected origin only.")
                st.plotly_chart(time_chart(visible, "predicted_power", "Normalized power [0, 1]", 390), width="stretch", config={"displaylogo": False})
                render_insights(summary, turbines)
                section_label("02", "ARCHIVED WEATHER INPUTS")
                st.subheader("The weather behind this forecast")
                st.caption("100 m wind is used as a hub-height proxy because turbine hub height was not provided.")
                columns = st.columns(2)
                for column, value, title in zip(columns, ("wind_speed_100m_ms", "temperature_2m_c"), ("100m wind speed · m/s", "Temperature · °C")):
                    with column:
                        st.markdown(f"**{title}**")
                        if value in visible and pd.api.types.is_numeric_dtype(visible[value]) and np.isfinite(visible[value]).all():
                            st.plotly_chart(time_chart(visible, value, title), width="stretch", config={"displaylogo": False})
                        else:
                            st.info("Saved weather values are missing or invalid. Restore the complete forecast CSV to display this chart.")
                render_provenance(visible)
        else:
            st.info("Choose a saved forecast to view generation, weather, deterministic insights, and run provenance.")
    with agent_tab:
        render_execution(audit, origin)
        st.divider()
        render_analysis(analysis, origin)
    with replay_tab:
        render_replay(ARTIFACTS_DIR, canonical, audits, origin, run_id, frame, analysis)
    with system_tab:
        render_system(frame, metrics)
        render_validation(metrics)
    st.caption("WindAgent AI · archived weather, deterministic power forecasts, auditable decisions. No forecast values are modified for display; KPI labels are rounded for readability.")
