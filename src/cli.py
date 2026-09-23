"""Reproducible commands for preparation, archive verification, and backtests."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    SCADA_CIVIL_TIMEZONE,
    SCADA_FIXED_UTC_OFFSET_HOURS,
    SCADA_TIMESTAMP_MODE,
    TURBINES,
)
from .diagnostics import persist_frame, timezone_alignment_metrics, weather_scada_diagnostics
from .evaluation import (
    build_weather_target_frame,
    daily_origins,
    persist_predictions,
    run_historical_walk_forward,
    run_walk_forward,
)
from .scada import aggregate_hourly, load_all_scada
from .training_data import (
    TIMEZONE_CANDIDATES,
    assess_timezone_evidence,
    build_archived_weather_rows,
    build_manifest,
    get_training_rows_before,
    hourly_scada_for_candidate,
    join_archived_weather_to_scada,
    lag_correlation_table,
    phase2a_artifact_paths,
    prepare_power_curve_inputs,
    timezone_alignment_table,
    to_training_schema,
    weather_height_comparison,
    write_json,
    write_training_dataset,
)
from .weather import (
    OpenMeteoSingleRunsClient,
    WeatherArchiveError,
    ensure_utc,
    select_latest_available_run,
)


def _hourly_scada(
    timestamp_mode: str = SCADA_TIMESTAMP_MODE,
    fixed_utc_offset_hours: float | None = SCADA_FIXED_UTC_OFFSET_HOURS,
) -> pd.DataFrame:
    return aggregate_hourly(
        load_all_scada(
            timestamp_mode=timestamp_mode,
            fixed_utc_offset_hours=fixed_utc_offset_hours,
        )
    )


def _hourly_scada_from_args(args: argparse.Namespace) -> pd.DataFrame:
    return _hourly_scada(args.scada_timestamp_mode, args.fixed_utc_offset_hours)


def command_prepare(_: argparse.Namespace) -> int:
    hourly = _hourly_scada()
    print("Hourly SCADA preparation (no interpolation)")
    if SCADA_TIMESTAMP_MODE == "civil_time":
        print(f"SCADA timestamp interpretation: civil_time ({SCADA_CIVIL_TIMEZONE}) -> UTC")
    else:
        print(
            "SCADA timestamp interpretation: "
            f"fixed_offset UTC{SCADA_FIXED_UTC_OFFSET_HOURS:+g} -> UTC"
        )
    print(
        hourly.groupby("turbine_id").agg(
            rows=("timestamp", "size"),
            complete_hours=("is_complete_hour", "sum"),
            incomplete_hours=("is_complete_hour", lambda series: (~series).sum()),
            empty_hours=("coverage_count", lambda series: (series == 0).sum()),
            partial_hours=("coverage_count", lambda series: ((series > 0) & (series < 6)).sum()),
        )
    )
    return 0


def command_smoke(_: argparse.Namespace) -> int:
    client = OpenMeteoSingleRunsClient()
    origins = ["2026-01-31T00:00:00Z", "2026-01-15T00:00:00Z", "2025-12-15T00:00:00Z"]
    print(
        "Open-Meteo Single Runs smoke test "
        f"(model=ecmwf_ifs, conservative availability lag={client.availability_lag_hours}h)"
    )
    for origin_text in origins:
        origin = ensure_utc(origin_text)
        selected, cutoff = select_latest_available_run(origin, client.availability_lag_hours)
        for turbine in TURBINES.values():
            forecast = client.get_forecast(
                turbine.latitude, turbine.longitude, origin, horizon_hours=48
            )
            data = forecast.data
            assert selected <= cutoff
            assert len(data) == 48
            assert data["valid_time_utc"].max() >= origin + pd.Timedelta(48, unit="h")
            print(
                f"{turbine.turbine_id} origin={origin.isoformat()} run={selected.isoformat()} "
                f"cutoff={cutoff.isoformat()} valid={data['valid_time_utc'].min().isoformat()}.."
                f"{data['valid_time_utc'].max().isoformat()} "
                f"wind100={data['wind_speed_100m'].min():.1f}-{data['wind_speed_100m'].max():.1f}m/s "
                f"temp={data['temperature_2m'].min():.1f}-{data['temperature_2m'].max():.1f}C"
            )
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    hourly = _hourly_scada_from_args(args)
    origins = daily_origins(args.start_date, args.end_date or args.start_date)
    predictions, metrics = run_walk_forward(
        hourly,
        origins,
        OpenMeteoSingleRunsClient(),
        training_days=args.training_days,
        horizon_hours=48,
    )
    path = persist_predictions(predictions)
    print(f"Persisted predictions: {path}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


def _parse_offsets(value: str) -> list[float]:
    offsets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not offsets:
        raise ValueError("At least one SCADA UTC offset must be supplied")
    return offsets


def command_phase2_diagnose(args: argparse.Namespace) -> int:
    """Build only pre-February diagnostic labels and test explicit clock offsets."""
    origins = daily_origins(args.start, args.end)
    client = OpenMeteoSingleRunsClient()
    candidate_tables: dict[str, pd.DataFrame] = {
        "civil_time_Asia_Almaty": build_weather_target_frame(
            _hourly_scada("civil_time"), origins, client
        )
    }
    for offset_hours in _parse_offsets(args.offsets):
        hourly = _hourly_scada("fixed_offset", offset_hours)
        candidate = f"fixed_offset_UTC{offset_hours:+g}"
        candidate_tables[candidate] = build_weather_target_frame(hourly, origins, client)

    alignment = timezone_alignment_metrics(candidate_tables)
    persist_frame(alignment, ARTIFACTS_DIR / "phase2_timezone_alignment.csv")
    requested_candidate = args.diagnostic_candidate
    if requested_candidate not in candidate_tables:
        raise ValueError(
            "--diagnostic-candidate must name a configured explicit candidate; "
            "no timezone is changed automatically"
        )
    diagnostic = candidate_tables[requested_candidate]
    weather_metrics, distributions = weather_scada_diagnostics(diagnostic)
    persist_frame(diagnostic, ARTIFACTS_DIR / "phase2_weather_scada_diagnostic_table.csv")
    persist_frame(weather_metrics, ARTIFACTS_DIR / "phase2_weather_scada_metrics.csv")
    persist_frame(distributions, ARTIFACTS_DIR / "phase2_weather_scada_distributions.csv")

    print("Timezone-alignment candidates (ECMWF 100m vs SCADA wind; all pre-February):")
    print(
        alignment.loc[alignment["scope"].eq("ALL")]
        .sort_values(["correlation", "mae"], ascending=[False, True])
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nWeather diagnostic uses timestamp interpretation={requested_candidate}")
    print(
        weather_metrics.loc[
            weather_metrics["lead_time_hours"].eq("overall")
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nDiagnostic labelled rows: {diagnostic['actual_scada_power'].notna().sum():,}")
    return 0


def command_phase2_backtest(args: argparse.Namespace) -> int:
    """Build archived-weather samples and run daily pre-February walk-forward tests."""
    hourly = _hourly_scada_from_args(args)
    origins = daily_origins(args.training_start, args.validation_end)
    table = build_weather_target_frame(hourly, origins, OpenMeteoSingleRunsClient())
    table_path = persist_frame(table, ARTIFACTS_DIR / "phase2_historical_forecast_training_table.csv")
    validation = daily_origins(args.validation_start, args.validation_end)
    predictions, metrics, errors = run_historical_walk_forward(
        hourly,
        table,
        validation,
        catboost_iterations=args.catboost_iterations,
    )
    prediction_path = persist_predictions(predictions, ARTIFACTS_DIR / "phase2_walk_forward_predictions.csv")
    metric_path = persist_frame(metrics, ARTIFACTS_DIR / "phase2_walk_forward_metrics.csv")
    error_path = persist_frame(errors, ARTIFACTS_DIR / "phase2_error_analysis.csv")
    training_samples = int(table["actual_scada_power"].notna().sum())
    print(
        f"Historical archived-weather labelled samples: {training_samples:,}; "
        f"table={table_path}"
    )
    print(f"predictions={prediction_path}; metrics={metric_path}; errors={error_path}")
    print("\nHeld-out overall metrics (all turbines):")
    print(
        metrics.loc[
            metrics["turbine_id"].eq("ALL") & metrics["horizon_band"].eq("overall")
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    return 0


def command_phase2a_build(args: argparse.Namespace) -> int:
    """Build diagnostics and an archived-weather dataset without model fitting."""
    origins = daily_origins(args.start_date, args.end_date)
    if origins.empty:
        raise ValueError("No daily Almaty-midnight forecast origins were generated")
    print(
        "Phase 2A: retrieving one pinned archived forecast per turbine/origin "
        f"for {len(origins)} daily origins.",
        flush=True,
    )
    archive = build_archived_weather_rows(
        origins,
        OpenMeteoSingleRunsClient(),
        horizon_hours=48,
        progress=lambda message: print(message, flush=True),
    )
    diagnostic_origins = daily_origins(
        args.diagnostic_start_date, args.diagnostic_end_date
    )
    diagnostic_origin_set = set(diagnostic_origins)
    diagnostic_weather = archive.frame.loc[
        archive.frame["forecast_origin_utc"].isin(diagnostic_origin_set)
    ].copy()
    if diagnostic_weather.empty:
        raise ValueError("The requested diagnostic dates are outside the archive build range")
    candidate_hourly = {
        candidate: hourly_scada_for_candidate(candidate)
        for candidate in TIMEZONE_CANDIDATES
    }
    candidate_full_joined = {
        candidate: join_archived_weather_to_scada(archive.frame, hourly)
        for candidate, hourly in candidate_hourly.items()
    }
    candidate_diagnostic_joined = {
        candidate: join_archived_weather_to_scada(diagnostic_weather, hourly)
        for candidate, hourly in candidate_hourly.items()
    }
    alignment = timezone_alignment_table(candidate_diagnostic_joined)
    decision = assess_timezone_evidence(alignment)
    selected_timezone = str(decision["selected_timezone_mode"])
    selected_hourly = candidate_hourly[selected_timezone]
    selected_diagnostic_joined = candidate_diagnostic_joined[selected_timezone]
    selected_full_joined = candidate_full_joined[selected_timezone]
    lag_scan = lag_correlation_table(diagnostic_weather, selected_hourly, selected_timezone)
    height = weather_height_comparison(selected_diagnostic_joined, selected_timezone)
    training = to_training_schema(selected_full_joined)

    february_start = pd.Timestamp("2026-02-01T00:00:00Z")
    if training["valid_time_utc"].ge(february_start).any():
        raise ValueError("Phase 2A training build would include February 2026 target timestamps")
    # Build OOF-only curve preparations. They are diagnostic artefacts, not a
    # tuned model or a model-selection decision.
    power_curve_rows, power_curve_summary = prepare_power_curve_inputs(training)
    oof_columns = [
        "turbine_id",
        "forecast_origin_utc",
        "valid_time_utc",
        "calibrated_wind_estimate_oof_ms",
        "power_curve_direct_oof_pred",
        "power_curve_calibrated_oof_pred",
    ]
    training = training.merge(
        power_curve_rows.loc[:, oof_columns],
        on=["turbine_id", "forecast_origin_utc", "valid_time_utc"],
        how="left",
        validate="one_to_one",
    )
    # Exercise the future outer-fold interface once without using an outer
    # period's labels as model inputs.
    _ = get_training_rows_before(training, training["valid_time_utc"].max() + pd.Timedelta(1, unit="h"))

    paths = phase2a_artifact_paths()
    for name, path in paths.items():
        if name != "dataset_dir":
            path.parent.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(paths["alignment_csv"], index=False)
    lag_scan.to_csv(paths["lag_csv"], index=False)
    height.to_csv(paths["height_csv"], index=False)
    power_curve_summary.to_csv(paths["power_curve_csv"], index=False)
    write_json(
        {
            "diagnostic_forecast_origin_date_range": {
                "start": args.diagnostic_start_date,
                "end": args.diagnostic_end_date,
            },
            "timezone_candidates": list(TIMEZONE_CANDIDATES),
            "decision": decision,
            "rows": alignment.to_dict(orient="records"),
        },
        paths["alignment_json"],
    )
    write_json(
        {
            "selected_timezone_mode": selected_timezone,
            "height_comparison": height.to_dict(orient="records"),
            "lag_scan_file": str(paths["lag_csv"]),
        },
        paths["weather_scada_json"],
    )
    dataset_path = write_training_dataset(training, paths["dataset_dir"])
    manifest = build_manifest(training, archive, selected_timezone, selected_hourly, dataset_path)
    manifest["timezone_evidence"] = decision
    manifest["power_curve_preparation_file"] = str(paths["power_curve_csv"])
    write_json(manifest, paths["manifest"])

    overall = alignment.loc[
        alignment["turbine"].eq("ALL") & alignment["lead_group"].eq("overall")
    ].sort_values("timezone_mode")
    best_lag = lag_scan.loc[lag_scan["turbine"].eq("ALL")].sort_values(
        "wind_pearson_corr", ascending=False
    ).iloc[0]
    best_height = height.loc[height["turbine"].eq("ALL")].sort_values(
        "wind_pearson_corr", ascending=False
    ).iloc[0]
    print("\nTimezone alignment, all turbines / all leads:")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nDecision: {selected_timezone}; evidence={decision['evidence']}. "
        f"{decision['reason']}"
    )
    print(
        f"Lag diagnostic peak: {int(best_lag['lag_hours']):+d}h, "
        f"r={best_lag['wind_pearson_corr']:.4f}, n={int(best_lag['n'])}."
    )
    print(
        f"Height diagnostic peak: {int(best_height['weather_wind_height_m'])}m, "
        f"r={best_height['wind_pearson_corr']:.4f}, n={int(best_height['n'])}."
    )
    print(
        f"\nDataset: {dataset_path}; rows={len(training):,}; "
        f"origins={archive.forecast_origins}; cache hits={archive.cache_hits}, misses={archive.cache_misses}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="prepare and report hourly SCADA")
    prepare.set_defaults(func=command_prepare)
    smoke = subcommands.add_parser("smoke", help="verify real archived-weather retrieval")
    smoke.set_defaults(func=command_smoke)
    backtest = subcommands.add_parser("backtest", help="run leakage-safe walk-forward backtest")
    backtest.add_argument(
        "--start-date",
        default="2026-01-29",
        help="civil date under the configured almaty_midnight origin convention",
    )
    backtest.add_argument("--end-date", help="optional inclusive civil end date")
    backtest.add_argument("--training-days", type=int, default=14)
    _add_scada_timestamp_arguments(backtest)
    backtest.set_defaults(func=command_backtest)

    diagnose = subcommands.add_parser(
        "phase2-diagnose", help="pre-February weather/SCADA and timezone diagnostics"
    )
    diagnose.add_argument("--start", default="2025-12-01T00:00:00Z")
    diagnose.add_argument("--end", default="2026-01-30T00:00:00Z")
    diagnose.add_argument("--offsets", default="0,4,5,6")
    diagnose.add_argument(
        "--diagnostic-candidate",
        default="civil_time_Asia_Almaty",
        help="one of civil_time_Asia_Almaty or a requested fixed_offset_UTC+Nh candidate",
    )
    diagnose.set_defaults(func=command_phase2_diagnose)

    phase2 = subcommands.add_parser(
        "phase2-backtest", help="daily expanding-window archived-weather model comparison"
    )
    phase2.add_argument("--training-start", default="2025-10-01T00:00:00Z")
    phase2.add_argument("--validation-start", default="2025-12-01T00:00:00Z")
    phase2.add_argument("--validation-end", default="2026-01-30T00:00:00Z")
    _add_scada_timestamp_arguments(phase2)
    phase2.add_argument("--catboost-iterations", type=int, default=250)
    phase2.set_defaults(func=command_phase2_backtest)

    phase2a = subcommands.add_parser(
        "phase2a-build",
        help="build pre-February timezone diagnostics and archived-weather training data",
    )
    phase2a.add_argument("--start-date", default="2025-10-01")
    phase2a.add_argument("--end-date", default="2026-01-29")
    phase2a.add_argument("--diagnostic-start-date", default="2025-12-01")
    phase2a.add_argument("--diagnostic-end-date", default="2026-01-29")
    phase2a.set_defaults(func=command_phase2a_build)
    development = subcommands.add_parser("phase2b-backtest", help="origin-safe four-model walk-forward comparison")
    development.add_argument("--start-date", default="2025-12-01")
    development.add_argument("--end-date", default="2026-01-29")
    development.set_defaults(func=command_development)
    acceptance = subcommands.add_parser("accept-validation", help="run tests/archive checks and persist an explicit validation review")
    acceptance.add_argument("--review", required=True, help="scientific review of held-out quality and limitations")
    acceptance.set_defaults(func=command_acceptance)
    freeze = subcommands.add_parser("freeze-models", help="freeze selected pre-February turbine models")
    freeze.set_defaults(func=command_freeze)
    february = subcommands.add_parser("february-forecast", help="frozen-model rolling archived forecasts; no SCADA")
    february.set_defaults(func=command_february)
    agent = subcommands.add_parser("agent-forecast", help="audited tool-based forecast; naive origin is Asia/Almaty")
    agent.add_argument("--origin", required=True)
    agent.add_argument("--turbines", default="T1,T2")
    agent.add_argument("--no-llm", action="store_true")
    agent.add_argument("--cache-only", action="store_true")
    agent.set_defaults(func=command_agent)
    replay = subcommands.add_parser("replay-february", help="replay 29 origins through the orchestrator, cache-only/no-LLM")
    replay.set_defaults(func=command_replay)
    return parser


def command_development(args: argparse.Namespace) -> int:
    from .development import run_model_development
    run_model_development(args.start_date, args.end_date)
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    from .development import freeze_final_models
    freeze_final_models()
    return 0


def command_acceptance(args: argparse.Namespace) -> int:
    import json
    import re
    import subprocess
    from .archive_audit import verify_training_archive
    from .development import accept_validation
    from .config import PROJECT_ROOT
    tested = subprocess.run([sys.executable, "-W", "ignore::DeprecationWarning", "-m", "pytest", "-q"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(tested.stdout)
    if tested.returncode:
        raise ValueError("Test suite failed; validation gate cannot pass")
    count = re.search(r"(\d+) passed", tested.stdout)
    if not count:
        raise ValueError("Could not verify passing test count")
    verify_training_archive()
    print(json.dumps(accept_validation(args.review, int(count.group(1))), indent=2))
    return 0


def command_february(args: argparse.Namespace) -> int:
    from .forecasting import generate_february
    generate_february()
    return 0


def command_agent(args: argparse.Namespace) -> int:
    from .agent import ForecastOrchestrator
    result = ForecastOrchestrator(client=OpenMeteoSingleRunsClient(cache_only=args.cache_only)).run(
        args.origin, tuple(part.strip() for part in args.turbines.split(",")), no_llm=args.no_llm)
    print(f"Forecast saved: {result.path}")
    return 0


def command_replay(args: argparse.Namespace) -> int:
    from .agent import replay_february
    replay_february()
    return 0


def _add_scada_timestamp_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scada-timestamp-mode",
        choices=("civil_time", "fixed_offset"),
        default=SCADA_TIMESTAMP_MODE,
        help="explicit interpretation of timezone-naive SCADA timestamps",
    )
    parser.add_argument(
        "--fixed-utc-offset-hours",
        type=float,
        default=SCADA_FIXED_UTC_OFFSET_HOURS,
        help="required only with --scada-timestamp-mode=fixed_offset",
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except WeatherArchiveError as error:
        print(f"Weather archive verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
