"""Reproducible commands for preparation, archive verification, and backtests."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import DEFAULT_AVAILABILITY_LAG_HOURS, TURBINES
from .evaluation import persist_predictions, run_walk_forward
from .scada import aggregate_hourly, load_all_scada
from .weather import (
    OpenMeteoSingleRunsClient,
    WeatherArchiveError,
    ensure_utc,
    select_latest_available_run,
)


def _hourly_scada() -> pd.DataFrame:
    return aggregate_hourly(load_all_scada())


def command_prepare(_: argparse.Namespace) -> int:
    hourly = _hourly_scada()
    print("Hourly SCADA preparation (no interpolation)")
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
            assert data["target_timestamp"].max() >= origin + pd.Timedelta(48, unit="h")
            print(
                f"{turbine.turbine_id} origin={origin.isoformat()} run={selected.isoformat()} "
                f"cutoff={cutoff.isoformat()} valid={data['target_timestamp'].min().isoformat()}.."
                f"{data['target_timestamp'].max().isoformat()} "
                f"wind100={data['wind_speed_100m'].min():.1f}-{data['wind_speed_100m'].max():.1f}m/s "
                f"temp={data['temperature_2m'].min():.1f}-{data['temperature_2m'].max():.1f}C"
            )
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    hourly = _hourly_scada()
    origins = [ensure_utc(value.strip()) for value in args.origins.split(",") if value.strip()]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="prepare and report hourly SCADA")
    prepare.set_defaults(func=command_prepare)
    smoke = subcommands.add_parser("smoke", help="verify real archived-weather retrieval")
    smoke.set_defaults(func=command_smoke)
    backtest = subcommands.add_parser("backtest", help="run leakage-safe walk-forward backtest")
    backtest.add_argument(
        "--origins", default="2026-01-29T00:00:00Z", help="comma-separated UTC origins"
    )
    backtest.add_argument("--training-days", type=int, default=14)
    backtest.set_defaults(func=command_backtest)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except WeatherArchiveError as error:
        print(f"Weather archive verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
