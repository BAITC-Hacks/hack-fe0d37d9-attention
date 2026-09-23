# Leakage-safe wind power forecasting

Hourly 24-48 hour normalized-power forecasting for two wind turbines. Phases
2B-4 are implemented and executed: pre-February walk-forward comparison,
frozen-model February inference, and an auditable tool orchestrator with
optional OpenAI analysis. No dashboard or decorative multi-agent framework.

## Data, locations, and SCADA time

The original 10-minute SCADA CSV files are read in place and never modified.
The pipeline stores timezone-aware UTC timestamps internally, aggregates to
hours without interpolation, and preserves partial and empty hours for
diagnostics. A training target must have six unique timestamps at the expected
00, 10, 20, 30, 40, and 50 minute cadence.

| Turbine | Latitude | Longitude |
|---|---:|---:|
| T1 | 43.645150 | 78.535604 |
| T2 | 43.643198 | 78.538828 |

The source SCADA timestamps have no timezone marker. The default
`SCADA_TIMESTAMP_MODE=civil_time` applies `Asia/Almaty` historical IANA rules,
then converts to UTC. `fixed_offset` is available only with an explicitly
configured `SCADA_FIXED_UTC_OFFSET_HOURS`; neither option proves the organizer
source-clock convention.

Kazakhstan's 2024-03-01 UTC+06 to UTC+05 transition creates an ambiguous local
23:00 hour. The raw data records that hour once, so the civil-time policy maps
it to the earlier UTC+06 occurrence. This is documented rather than inferred;
timezone-alignment diagnostics must still compare civil time with explicit
fixed-offset candidates before model selection.

Hub height was not supplied. `wind_speed_100m` is a modelling proxy, not a
known turbine specification.

Inputs are the two original 10-minute CSVs: ID, naive statistical timestamp,
wind speed, normalized active power (0-1), and ambient temperature. The
pre-February history begins in March 2023. February SCADA is never consumed by
model development or inference; these phases use only the guarded Phase 2A
pre-February dataset. Neither weather observations nor reanalysis are predictors.

## Point-in-time weather contract

Weather is retrieved only from Open-Meteo Single Runs, pinned to archived ECMWF
IFS HRES (`ecmwf_ifs`). Requests explicitly set `wind_speed_unit=ms`,
`timezone=UTC`, and `timeformat=unixtime`. No observations, reanalysis, or
un-pinned provider model may substitute for a missing archived run.

Historical **observations** describe what actually happened. Reanalysis blends
observations and later processing. Historical **forecasts** describe predictions
issued in the past. A Single Run pins one model initialization, unlike a stitched
archive that may mix issue times. Initialization is not publication: the seven-hour
lag is an explicit conservative assumption, not verified publication telemetry.
See [Open-Meteo Single Runs documentation](https://open-meteo.com/en/docs/single-runs-api).

The default availability lag is seven hours. For every sample, the following
invariants are checked:

- `weather_run_init_utc + availability_lag <= forecast_origin_utc`;
- `weather_available_at_utc <= forecast_origin_utc`;
- `valid_time_utc > forecast_origin_utc` and lead time is 1 through 48 hours.

Each cache entry has raw response and metadata sidecars. Metadata records the
origin, run initialization, availability time, provider/model, coordinates,
request parameters, retrieval time, and raw-response SHA-256.

The hackathon forecast-origin convention is `00:00 Asia/Almaty` per day,
converted to UTC before retrieval and persistence. It is not confused with
00:00 UTC.

## Leakage rules and modelling status

SCADA wind, temperature, and power are historical labels or post-forecast
diagnostics only. They are not forecast-time features. At outer origin `O`, a
training sample requires both `forecast_origin_utc < O` and
`valid_time_utc < O`; a past-origin forecast whose target lies after `O` is
rejected.

`TemporalLeakageGuard` also requires aware, nonmissing clocks, consistent lead
features, and completed training hours (`valid_time + 1 hour <= origin`). Joins
are exact UTC joins, never nearest-time. Missing target hours remain missing.
Gap utilities reindex the hourly timeline and segment it before any future
rolling calculations; primary predictors do not use target lags.

High-wind zero-power records are retained in operational backtest metrics. They
are flagged as `zero_power_high_wind` and `suspected_unavailability`; only the
latter is excluded from model and power-curve fitting. Empirical power curves
are separate for T1 and T2. If a power-curve prediction is used as a model
feature, it must be generated through chronological expanding-window OOF
predictions, never in sample.

The original Phase 1 accuracy figures predate the temporal audit and are not
valid evidence for model choice. Phase 2B now compares four fixed strategies
on daily December/January origins before any February inference.

## Phase 2A archived-forecast dataset

Phase 2A uses daily 00:00 Asia/Almaty origins from 2025-10-01 through
2026-01-29. It produced 11,616 archived ECMWF forecast rows, with 48 leads for
each of two turbines and 121 origins. The CSV fallback is used because no
Parquet engine is installed.

Timezone diagnostics use only December 2025-January 2026 exact UTC matches.
Civil time and fixed UTC+05:00 are identical during this period. Fixed UTC+06:00
has a small, mixed wind difference but better temperature alignment, so the
result is inconclusive and the configured civil-time default is retained. The
diagnostic never changes production configuration automatically.

Wind correlations were approximately 0.7025 at 10m and 0.7194 at 80/100/120m.
These diagnostics do not establish a physical hub height; 100m remains the
documented primary proxy. Other levels remain available as ML inputs.

The data artifact contains archived weather, calendar features, target labels,
coverage flags, and OOF-only direct/calibrated power-curve preparation fields.
Power-curve OOF summaries are diagnostic preparation, not a model-selection
result. Generated files are under `artifacts/datasets/` and
`artifacts/diagnostics/`.

Phase 2B corrected the OOF preparation to use forecast-origin cutoffs, not only
target-time blocks. Old Phase 2A OOF columns are ignored by model development.
The new `origin_oof_curve_features.csv` records each curve fit's maximum label
time. Curve fitting de-duplicates realised SCADA hours across forecast origins.

The four strategies are separate turbine-specific empirical curves, forecast
wind calibrated with HistGradientBoosting then passed through the curves,
direct HistGradientBoosting power regression, and separate hybrid CatBoost
models using origin-safe OOF curve predictions. CatBoost has one deterministic
configuration (450 trees, depth 6, learning rate 0.05, RMSE loss).

Primary metrics include every complete observed hour, including suspected
unavailability. Partial 48-hour target folds are explicitly marked and a
full-fold-only comparison is saved alongside the primary comparison. Model
selection uses held-out MAE, with a predeclared 0.002 simplicity tie tolerance.

## Executed model comparison and selection

Validation uses **60 daily origins**: December 1-31 and January 1-29, local
midnight. All four candidates refit using only targets and forecast origins
strictly earlier than the current outer origin. No random split, early-stopping
random validation, February tuning, or large hyperparameter search.
Monthly reports are grouped by the local **forecast-origin** month, not the
target month; a December 31 forecast can therefore score January 1 targets.

| Strategy | December MAE | January MAE | Combined MAE | Combined RMSE | Combined R² |
|---|---:|---:|---:|---:|---:|
| A: direct ECMWF 100m → SCADA empirical curve | 0.222549 | 0.157537 | **0.190973** | 0.282276 | 0.398645 |
| B: forecast weather → calibrated wind → curve | 0.214218 | 0.168369 | 0.191950 | 0.280306 | 0.407009 |
| C: HistGradientBoosting → power | 0.230339 | 0.184247 | 0.207952 | 0.277975 | 0.416828 |
| D: hybrid CatBoost + chronological OOF curve | 0.236792 | 0.187941 | 0.213065 | 0.280565 | 0.405911 |

December N=2,948; January N=2,784; pooled N=5,732 scored forecast/target pairs
per strategy. Overlapping horizons intentionally score each issued forecast;
these are **not independent unique observed hours**. There are 112 complete
48-hour turbine folds and 8 explicitly partial target folds. The full-fold-only
comparison scores 5,376 pairs: A MAE=0.186336, B=0.192054, C=0.208450,
D=0.215211. Primary metrics include suspected unavailability.

**Selected: A, separate empirical curves for T1 and T2.** It minimizes held-out
MAE and is simplest within the declared tie tolerance. It is transparent, but
does not eliminate forecast-to-SCADA domain shift. B has much smaller bias and
better December/worst-day behavior; C has slightly better RMSE. Neither has lower
pooled MAE. No claim that the selected baseline is best for every operating cost.

| Selected A, combined months | N | MAE | RMSE | R² | Capacity error, pp |
|---|---:|---:|---:|---:|---:|
| T1, 1-48h | 2,872 | 0.189175 | 0.279353 | 0.415544 | 18.918 |
| T2, 1-48h | 2,860 | 0.192779 | 0.285181 | 0.381131 | 19.278 |
| Both, 1-24h | 2,866 | 0.184555 | 0.272472 | 0.439688 | 18.456 |
| Both, 25-48h | 2,866 | 0.197391 | 0.291751 | 0.357572 | 19.739 |

Pooled MAE is 19.097 percentage points of rated normalized capacity. Important
failure pattern: negative residual bias (-0.114151). At actual power >=0.95,
T1/T2 mean residuals are approximately -0.306/-0.324. At actual power <=0.02,
MAE is about 0.055/0.056. Longer leads are worse. All models have saved
breakdowns by lead, turbine, forecast wind bucket, generation bucket, local
hour, month, near-rated output, and very low generation. No training metric
was used to establish these results.

The original Phase 2A OOF scores are superseded for selection: an audit found
that target-block cutoffs could precede a block's target times yet follow its
historical forecast origin. This was fixed; the new hybrid uses origin-safe
OOF features with a seven-day warm-up. No old OOF predictions are reused.

## Frozen models and February outputs

Model version: `34979ef7f8ba489c9c4ee7391c51bfbc`. Training cutoff is
**2026-01-30 19:00 UTC = January 31 00:00 Asia/Almaty**, before the first
transition forecast. Maximum training target timestamp is 2026-01-30 18:00 UTC.
The frozen fit uses 11,358 eligible forecast-origin rows; realised SCADA hours
are deduplicated when fitting each empirical curve. T1/T2 joblib files include
the fitted curve; model hashes, feature schema, seed, parameters, dataset hash,
package versions, and cutoff are recorded in the versioned manifest.

Origins run **January 31 through February 28**, each at 00:00 Asia/Almaty.
Including January 31 supplies the transition forecast for February 1. Each
origin predicts leads 1..48; the final origin intentionally extends into March.
The model never retrains on February actuals. Updated eligible weather produces
a fresh forecast with the same frozen model.

- `artifacts/predictions/february_rolling_forecast.csv`: 29 origins × 2 turbines × 48 = **2,784 rows**.
- `artifacts/predictions/february_submission.csv`: **2,686 rows** whose valid times fall in local February; forecast origins and overlapping leads are retained, not averaged or deduplicated.
- `artifacts/diagnostics/february_forecast_qc.json`: all horizons, clocks, availability, finite/bounded predictions passed.
- `artifacts/diagnostics/february_forecast_summaries.json`: deterministic 24h/48h means, peaks, lows, three-hour minimum blocks, hourly ramps, and turbine differences.

No fake `y_true` column or February accuracy metric is provided. Immutable
per-run forecast/provenance/summary copies remain under `artifacts/predictions/runs/`.

## One orchestrator and safe tools

```text
context → archived weather → weather validation → features → frozen inference
        → power validation → deterministic summary → persistence → optional analysis
```

`src/agent.py` is one executable, deterministic state machine, not a collection
of pretend LLM agents. Typed tools in `src/forecasting.py` enforce every gate.
The orchestrator may retry retrieval, choose an older eligible run, stop on
invalid inputs, and recalculate; neither an LLM nor a prompt decides eligibility.

Operational decisions:

- Connection errors, HTTP 429, and 5xx: at most two retries with 1s/2s backoff per request.
- Missing run (404/410 or an explicit run-unavailable 400): try at most two older six-hour initializations. Every run passes the same seven-hour cutoff. Never a newer unavailable run, observations, or another model.
- Missing variables, wrong units, invalid provenance, or invalid predictions: fail, record the error, do not publish a forecast. No synthetic fills.
- Recalculation: new `run_id`, preserve prior files, record previous run and whether weather response hashes changed.
- LLM failure: retain the validated forecast and record an optional-analysis warning.

Each `artifacts/agent_runs/<run_id>.json` records start/end times, context,
tool order/status/timing, weather cache/retry/fallback decisions, selected
initializations, model hashes/version, output path, warnings, errors, and
recalculation lineage. Forecasts are not published before numerical validation.

OpenAI is optional. `OPENAI_API_KEY` is read from the environment or ignored
`.env`; it is never logged or saved. `OPENAI_MODEL` defaults to `gpt-4.1-mini`.
The analysis layer uses the Responses API with
[strict structured output](https://developers.openai.com/api/docs/guides/structured-outputs).
The LLM selects salient IDs from supplied deterministic facts, and Python
renders their sentences. This intentionally constrained MVP prevents invented
numbers and causal claims. It has no forecast dataframe or execution authority.
No key was available during the original Phase 4 run;
success, invalid output, and failure paths are unit-tested with mocks, at no cost.

### OpenAI audit metadata

New `artifacts/agent_runs/<run_id>.json` files include a top-level `openai` object
(also attached to the existing analysis result): response ID, returned model,
requested model, execution/API response statuses, server `created_at` when
provided, UTC request start/end, locally measured monotonic `latency_ms`, and
token usage. Only `x-request-id` is read from response headers for the optional
`openai_request_id`; full headers are not persisted. Field meanings follow the
[Responses schema](https://developers.openai.com/api/reference/python/resources/responses/methods/retrieve)
and [request-ID documentation](https://developers.openai.com/api/reference/overview#debugging-requests).

Missing token counts are `null`, never estimated. Nested usage retains only
allowlisted nonnegative aggregate counters (for example cached/reasoning tokens);
unknown fields are ignored. Missing or malformed usage does not fail analysis.
`called=true` means a request was attempted, not proof that a server received it
after a network failure. Skips record `called=false` and a reason. Failures keep
timings, any received IDs/usage, and a fixed safe error message, never raw provider
errors or stack traces. `response_status` preserves the API status separately
when local fact-selection validation fails. Uninstrumented legacy/custom adapters
record `called=null`; old audit files are left unchanged.

API keys are never persisted. Full LLM prompts, request bodies, authorization
headers, cookies, and hidden instructions are not stored. Prompt audit records
only analysis type, template version, and fact count; the existing run context
already supplies turbine IDs, forecast origin, and run ID. Schema-validated LLM
fact selection cannot change deterministic numerical forecasts, timestamps,
weather runs, warnings, model identity, horizons, or QC.

Audit-patch verification: 51 tests passed (12 additional mocked cases), compile
checks passed. Exactly one live call was executed on 2026-09-23 after the tests:
run `3e41a55d2a4d4b35a2cbac8db2371629`, HTTP 200, returned model
`gpt-4.1-mini-2025-04-14`, status `completed`, latency 3609 ms, usage 760 input +
29 output = 789 tokens. Response ID and request ID are recorded in that run's
audit. The forecast CSV matched the earlier run in every column except the new
`run_id`; no original audit files were rewritten. Tests use synthetic credentials
only, and no API credits. Existing NumPy timedelta deprecation warnings remain
outside this isolated audit patch.

`replay-february` always disables network and LLM calls. It replays all 29 origins
through the same orchestrator, saves new immutable run records and separate
`february_replay_*` exports, and checks exact equality with canonical predictions.
Executed result: **58 cache hits, zero network requests, identical predictions**.

## Structure and artifacts

```text
src/
  scada.py, gaps.py             SCADA quality and contiguous segments
  weather.py, leakage.py        archived requests and deterministic time guards
  features.py, training_data.py prediction features and exact target joins
  models.py, development.py     curves, OOF, four-model walk-forward, frozen fit
  archive_audit.py              hash/exact-value cache reconciliation
  forecasting.py               typed deterministic operational tools
  agent.py, analysis_layer.py   orchestration and optional grounded explanation
  cli.py                       reproducible commands
tests/                         unit tests, including mocked provider/OpenAI failures
data/cache/weather/            ignored raw archived weather and metadata
artifacts/
  datasets/                    forecast_training.csv, origin OOF features
  metrics/                     model_comparison.csv and .json
  diagnostics/                 errors, temporal audits, acceptance, forecast QC
  models/                      model_selection.json and versioned final models
  predictions/                 walk-forward, February, replay, immutable runs
  agent_runs/                  execution audit JSONs
```

The selection artifact is `artifacts/models/model_selection.json`.
`artifacts/diagnostics/error_analysis.csv` and
`artifacts/predictions/walk_forward_predictions.csv` retain the full comparison.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env

python -m pytest -q
python -m src.cli prepare
python -m src.cli smoke
python -m src.cli phase2a-build
python -m src.archive_audit
python -m src.cli phase2b-backtest

# Review the held-out metrics/errors before passing the acceptance gate.
# This command re-runs tests and reconciles every training forecast with cache.
python -m src.cli accept-validation --review "A has lowest pooled MAE and is simplest; B has lower bias. Keep high-generation underprediction as an explicit limitation."
python -m src.cli freeze-models
python -m src.cli february-forecast
python -m src.cli agent-forecast --origin "2026-02-10 00:00" --turbines T1,T2 --no-llm --cache-only
python -m src.cli agent-forecast --origin "2026-02-10 00:00" --cache-only
python -m src.cli replay-february
python -m compileall -q src tests
```

Do not overwrite an existing `.env` with the example. The second agent command
uses OpenAI only if a key is configured; `--cache-only` applies to weather, not
OpenAI. Use `--no-llm` to guarantee no OpenAI call. A naive CLI origin explicitly
means Asia/Almaty local time, while an offset-aware input is converted to UTC.

`phase2b-backtest --start-date 2025-12-15 --end-date 2025-12-15` is supported for
a small experiment, but **overwrites comparison artifacts and does not satisfy
the two-month gate**. Re-run the complete default comparison before selecting
or freezing models. The same applies to a January-only sample. The executed
60-origin run already includes both December and January samples.

No new mandatory dependency or dashboard framework was added. The existing
requirements include pandas, numpy, requests, sklearn, CatBoost, joblib, pytest,
and python-dotenv. Exact fitted-package versions are saved with the model.
`data/cache/`, `artifacts/`, `.env`, and Python caches remain ignored; original
SCADA CSVs were not changed.

Environment settings are in `.env.example`: `SCADA_TIMESTAMP_MODE`,
`SCADA_FIXED_UTC_OFFSET_HOURS`, `FORECAST_ORIGIN_CONVENTION`,
`WEATHER_AVAILABILITY_LAG_HOURS`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
`OPEN_METEO_API_KEY` is an unused placeholder: this client uses the public
non-commercial Single Runs endpoint and does not attach that key. Changing
timezone or availability policy requires new validation/model artifacts;
the orchestrator rejects a weather lag differing from the frozen manifest.

## Executed verification and limitations

39 tests passed, 0 failed; compile checks passed. Full December/January
walk-forward, archive reconciliation (242 cached requests/11,616 exact weather
matches), six weather smoke cases, model freezing, February generation, no-LLM
demo, no-key demo, and complete cache-only replay were executed successfully.
Synthetic data is used only in isolated unit fixtures, never in saved operational
or validation artifacts. Fallback and network-failure branches were mocked;
the real February runs required no fallback.

Unresolved assumptions: organizer timestamp convention and actual hub height;
seven-hour publication lag; negligible SCADA ingestion latency after each
hour finishes; grid-to-turbine spatial/height mismatch. Validation covers only
two winter months and the scored pairs overlap. January origins stop at the
29th. There is no uncertainty calibration or February ground-truth evaluation.
The final curve's high-output underprediction is material, not hidden by the
overall score. B is a reasonable future challenger if bias/RMSE matter more
than MAE; changing that objective requires a new documented pre-February review.

Before an LLM demo, configure a key and run one optional explanation explicitly.
Before a submission, confirm the organizer's required rolling-origin export
format. Future improvements: confirm clock/hub specifications, extend archived
training seasons, add chronological uncertainty calibration and an explicit
asymmetric operational loss, then build a dashboard over these immutable outputs.
