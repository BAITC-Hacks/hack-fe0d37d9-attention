"""Point-in-time Open-Meteo Single Runs client with an auditable disk cache."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import DATA_CACHE_DIR, DEFAULT_AVAILABILITY_LAG_HOURS
from .leakage import LeakageError, validate_weather_samples


SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
PROVIDER = "Open-Meteo Single Runs API"
MODEL = "ecmwf_ifs"
REQUESTED_VARIABLES = (
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_speed_120m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
)
EXPECTED_HOURLY_UNITS = {
    "time": "unixtime",
    "wind_speed_10m": "m/s",
    "wind_speed_80m": "m/s",
    "wind_speed_100m": "m/s",
    "wind_speed_120m": "m/s",
    "wind_direction_100m": "\u00b0",
    "temperature_2m": "\u00b0C",
    "surface_pressure": "hPa",
}


class WeatherArchiveError(RuntimeError):
    """Raised when an archived forecast cannot be retrieved or validated."""


class WeatherRunUnavailable(WeatherArchiveError):
    """Only this failure permits trying an older eligible initialization."""


def ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def select_latest_available_run(
    forecast_origin: pd.Timestamp | str,
    availability_lag_hours: int = DEFAULT_AVAILABILITY_LAG_HOURS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick the latest 00/06/12/18 UTC run safely before the availability cutoff.

    Open-Meteo documents that a global-model run can take about 4--6 hours to
    become available. The configurable default is deliberately more conservative:
    seven hours, so a run is never selected merely because it was initialised
    before the decision time.
    """
    if availability_lag_hours < 0:
        raise ValueError("availability_lag_hours must be non-negative")
    origin = ensure_utc(forecast_origin)
    cutoff = origin - pd.Timedelta(availability_lag_hours, unit="h")
    selected = cutoff.floor("6h")
    if selected > cutoff:  # Defensive invariant for any future frequency change.
        selected -= pd.Timedelta(6, unit="h")
    return selected, cutoff


@dataclass(frozen=True)
class WeatherMetadata:
    forecast_origin_utc: str
    weather_run_init_utc: str
    weather_available_at_utc: str
    availability_cutoff: str
    availability_lag_hours: int
    model: str
    provider: str
    latitude: float
    longitude: float
    requested_variables: list[str]
    retrieval_timestamp: str
    raw_response_sha256: str
    request_parameters: dict[str, Any]
    cache_key: str
    fallback_steps: int


@dataclass
class WeatherForecast:
    data: pd.DataFrame
    metadata: WeatherMetadata
    cache_hit: bool


class OpenMeteoSingleRunsClient:
    """Retrieve only explicitly pinned ECMWF IFS archived forecasts.

    There is intentionally no fallback to reanalysis, observations, best-match,
    or another weather model. Such a fallback would invalidate the backtest.
    """

    def __init__(
        self,
        cache_dir: Path = DATA_CACHE_DIR,
        availability_lag_hours: int = DEFAULT_AVAILABILITY_LAG_HOURS,
        timeout_seconds: int = 60,
        session: requests.Session | None = None,
        max_retries: int = 2,
        max_older_runs: int = 2,
        backoff_seconds: float = 1.0,
        cache_only: bool = False,
        sleeper=time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.availability_lag_hours = availability_lag_hours
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        if max_retries < 0 or max_older_runs < 0 or backoff_seconds < 0:
            raise ValueError("Retry settings must be nonnegative")
        self.max_retries = max_retries
        self.max_older_runs = max_older_runs
        self.backoff_seconds = backoff_seconds
        self.cache_only = cache_only
        self.sleeper = sleeper
        self.events: list[dict] = []

    def _event(self, action: str, **details) -> None:
        self.events.append(dict(action=action, at_utc=pd.Timestamp.now(tz="UTC").isoformat(), **details))

    def _request_parameters(
        self,
        latitude: float,
        longitude: float,
        origin: pd.Timestamp,
        selected_run: pd.Timestamp,
        horizon_hours: int,
    ) -> dict[str, str | int | float]:
        # A selected run can predate origin by 7--12 hours. Request enough model
        # lead time to include the full requested horizon, then filter locally.
        elapsed_hours = int(
            (origin - selected_run).total_seconds() // 3600
        )
        return {
            "latitude": round(float(latitude), 6),
            "longitude": round(float(longitude), 6),
            "models": MODEL,
            "run": selected_run.strftime("%Y-%m-%dT%H:%M"),
            "timezone": "UTC",
            "timeformat": "unixtime",
            "wind_speed_unit": "ms",
            "hourly": ",".join(REQUESTED_VARIABLES),
            "forecast_hours": elapsed_hours + horizon_hours + 1,
        }

    @staticmethod
    def _cache_key(parameters: dict[str, Any]) -> str:
        encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _paths(self, cache_key: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{cache_key}.json",
            self.cache_dir / f"{cache_key}.metadata.json",
        )

    def _load_or_request(
        self, parameters: dict[str, Any], cache_key: str
    ) -> tuple[bytes, dict[str, Any], bool]:
        raw_path, metadata_path = self._paths(cache_key)
        if raw_path.exists() and metadata_path.exists():
            self._event("cache_hit", cache_key=cache_key, run=parameters["run"])
            raw = raw_path.read_bytes()
            try:
                return raw, json.loads(raw.decode("utf-8")), True
            except (UnicodeDecodeError, ValueError) as error:
                raise WeatherArchiveError(
                    f"Cached archived weather response {raw_path.name} is invalid"
                ) from error

        self._event("cache_miss", cache_key=cache_key, run=parameters["run"])
        if self.cache_only:
            raise WeatherRunUnavailable("Archived run absent from cache; network disabled")
        for attempt in range(self.max_retries + 1):
            self._event("network_request", run=parameters["run"], attempt=attempt + 1)
            try:
                response = self.session.get(SINGLE_RUNS_URL, params=parameters, timeout=self.timeout_seconds)
            except requests.RequestException as error:
                if attempt == self.max_retries:
                    raise WeatherArchiveError("Archived weather network retries exhausted") from error
                self._event("retry", reason=type(error).__name__, delay_seconds=self.backoff_seconds * 2**attempt)
                self.sleeper(self.backoff_seconds * 2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    raise WeatherArchiveError(f"Archived weather HTTP {response.status_code}: retries exhausted")
                self._event("retry", reason=f"HTTP {response.status_code}", delay_seconds=self.backoff_seconds * 2**attempt)
                self.sleeper(self.backoff_seconds * 2**attempt)
                continue
            break
        if not response.ok:
            snippet = response.text[:500]
            explicit_missing = response.status_code == 400 and "run" in snippet.lower() and any(
                marker in snippet.lower() for marker in ("not found", "not available", "unavailable")
            )
            if response.status_code in (404, 410) or explicit_missing:
                raise WeatherRunUnavailable(f"Archived run {parameters['run']} unavailable (HTTP {response.status_code})")
            raise WeatherArchiveError(
                f"Open-Meteo did not provide archived run {parameters['run']} "
                f"(HTTP {response.status_code}): {snippet}"
            )
        raw = response.content
        try:
            payload = response.json()
        except ValueError as error:
            raise WeatherArchiveError("Open-Meteo response was not valid JSON") from error

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        # Metadata is written by get_forecast after validation. This partial file
        # is intentionally absent if parsing fails, so it cannot become a cache hit.
        return raw, payload, False

    @staticmethod
    def _parse_hourly(payload: dict[str, Any]) -> pd.DataFrame:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or "time" not in hourly:
            reason = payload.get("reason", "missing hourly time series")
            raise WeatherArchiveError(f"Open-Meteo response lacks hourly data: {reason}")
        frame = pd.DataFrame(hourly)
        expected = set(REQUESTED_VARIABLES)
        missing = expected.difference(frame.columns)
        if missing:
            raise WeatherArchiveError(
                f"Archived forecast omitted requested variables: {sorted(missing)}"
            )
        units = payload.get("hourly_units")
        if not isinstance(units, dict):
            raise WeatherArchiveError("Open-Meteo response lacks hourly unit metadata")
        wrong_units = {
            name: {"expected": expected, "received": units.get(name)}
            for name, expected in EXPECTED_HOURLY_UNITS.items()
            if units.get(name) != expected
        }
        if wrong_units:
            raise WeatherArchiveError(
                f"Open-Meteo response units do not match the persisted schema: {wrong_units}"
            )
        frame = frame.rename(columns={"time": "valid_time_utc"})
        raw_time = frame["valid_time_utc"]
        if pd.api.types.is_numeric_dtype(raw_time):
            frame["valid_time_utc"] = pd.to_datetime(
                raw_time, unit="s", errors="raise", utc=True
            )
        else:
            raise WeatherArchiveError(
                "Open-Meteo Single Runs response did not honor timeformat=unixtime"
            )
        for column in REQUESTED_VARIABLES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame["valid_time_utc"].duplicated().any():
            raise WeatherArchiveError("Archived forecast contains duplicate valid timestamps")
        return frame.sort_values("valid_time_utc").reset_index(drop=True)

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_origin: pd.Timestamp | str,
        horizon_hours: int = 48,
    ) -> WeatherForecast:
        """Return exactly the next `horizon_hours` valid UTC hours for one origin."""
        origin = ensure_utc(forecast_origin)
        selected_run, cutoff = select_latest_available_run(
            origin, self.availability_lag_hours
        )
        for step in range(self.max_older_runs + 1):
            run = selected_run - pd.Timedelta(6 * step, unit="h")
            try:
                forecast = self.get_forecast_from_run(latitude, longitude, origin, run, horizon_hours)
                if step:
                    self._event("older_run_selected", run=run.isoformat(), fallback_steps=step)
                return forecast
            except WeatherRunUnavailable:
                self._event("run_unavailable", run=run.isoformat(), fallback_steps=step)
                if step == self.max_older_runs:
                    raise
        raise AssertionError("Unreachable run-selection branch")

    def get_forecast_from_run(self, latitude: float, longitude: float,
                              forecast_origin: pd.Timestamp | str, selected_run: pd.Timestamp | str,
                              horizon_hours: int = 48) -> WeatherForecast:
        """Explicit replay run; never allows a caller to bypass the availability gate."""
        if not 1 <= horizon_hours <= 48:
            raise ValueError("horizon_hours must be between 1 and 48")
        origin = ensure_utc(forecast_origin)
        selected_run = ensure_utc(selected_run)
        latest, cutoff = select_latest_available_run(origin, self.availability_lag_hours)
        if origin != origin.floor("h") or selected_run != selected_run.floor("6h"):
            raise ValueError("Origin must be hourly and run must be a six-hour initialization")
        if selected_run > cutoff:
            raise LeakageError("Requested weather run is after availability cutoff")
        parameters = self._request_parameters(
            latitude, longitude, origin, selected_run, horizon_hours
        )
        cache_key = self._cache_key(parameters)
        raw, payload, cache_hit = self._load_or_request(parameters, cache_key)
        frame = self._parse_hourly(payload)

        first_target = origin + pd.Timedelta(1, unit="h")
        last_target = origin + pd.Timedelta(horizon_hours, unit="h")
        forecast = frame.loc[
            frame["valid_time_utc"].between(first_target, last_target, inclusive="both")
        ].copy()
        if len(forecast) != horizon_hours:
            raise WeatherArchiveError(
                f"Archived run {selected_run.isoformat()} does not cover {first_target} "
                f"through {last_target}; received {len(forecast)}/{horizon_hours} hours"
            )
        expected_index = pd.date_range(first_target, last_target, freq="1h", tz="UTC")
        if not forecast["valid_time_utc"].reset_index(drop=True).array.equals(
            pd.array(expected_index)
        ):
            raise WeatherArchiveError("Archived forecast valid times are not contiguous hourly UTC")
        if forecast[list(REQUESTED_VARIABLES)].isna().any().any():
            raise WeatherArchiveError("Archived forecast has missing requested weather values")

        available_at = selected_run + pd.Timedelta(self.availability_lag_hours, unit="h")
        if available_at > origin:
            raise AssertionError("Selected weather run was not available at forecast origin")
        raw_hash = hashlib.sha256(raw).hexdigest()
        metadata = WeatherMetadata(
            forecast_origin_utc=origin.isoformat(),
            weather_run_init_utc=selected_run.isoformat(),
            weather_available_at_utc=available_at.isoformat(),
            availability_cutoff=cutoff.isoformat(),
            availability_lag_hours=self.availability_lag_hours,
            model=MODEL,
            provider=PROVIDER,
            latitude=float(latitude),
            longitude=float(longitude),
            requested_variables=list(REQUESTED_VARIABLES),
            retrieval_timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
            raw_response_sha256=raw_hash,
            request_parameters=parameters,
            cache_key=cache_key,
            fallback_steps=int((latest - selected_run).total_seconds() / (6 * 3600)),
        )
        if not cache_hit:
            _, metadata_path = self._paths(cache_key)
            metadata_path.write_text(
                json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8"
            )
        else:
            # Cache metadata is persisted provenance, including the hash of the
            # exact raw response used for every repeated backtest invocation.
            _, metadata_path = self._paths(cache_key)
            cached = json.loads(metadata_path.read_text("utf-8"))
            if cached.get("raw_response_sha256") != raw_hash:
                raise WeatherArchiveError(
                    f"Cached weather provenance hash does not match {cache_key}"
                )
            if cached.get("request_parameters") != parameters or cached.get("provider") != PROVIDER or cached.get("model") != MODEL:
                raise WeatherArchiveError("Cached weather request provenance mismatch")
            # A cache entry describes the acquisition; availability/fallback
            # describe this invocation and must never be inherited from an old policy.
            metadata = replace(metadata, retrieval_timestamp=cached["retrieval_timestamp"])

        forecast["forecast_origin_utc"] = origin
        forecast["weather_run_init_utc"] = selected_run
        forecast["weather_available_at_utc"] = available_at
        forecast["availability_lag_hours"] = self.availability_lag_hours
        forecast["lead_time_hours"] = (
            (forecast["valid_time_utc"] - origin).dt.total_seconds() / 3600
        ).astype(int)
        forecast["model_lead_time_hours"] = (
            (forecast["valid_time_utc"] - selected_run).dt.total_seconds() / 3600
        ).astype(int)
        forecast["run_age_at_origin_hours"] = int(
            (origin - selected_run).total_seconds() / 3600
        )
        forecast["weather_model"] = MODEL
        validate_weather_samples(forecast)
        return WeatherForecast(data=forecast, metadata=metadata, cache_hit=cache_hit)
