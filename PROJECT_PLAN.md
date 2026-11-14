# Project Plan

## Project

**AI-Driven Windows OS Performance Prediction and Intelligent Resource Analysis**  
B.Tech CSE major project, UPES

## Research scope and dataset strategy

The project predicts future resource utilization on Windows using an **authentic, self-collected telemetry time series from the Windows machine on which collection runs**. It does not use generated/synthetic records as training evidence. The current data file is `data/raw/windows_telemetry.csv`.

This is a single-machine case study. Results describe this device and observed collection periods; they do not establish generalization to other Windows computers, hardware, users, or workloads. Each collection invocation has a session ID; feature and target windows must not cross session boundaries.

The Google Cluster Data 2011-2 plan is retained as a pending/abandoned alternative because official access approval was taking too long. It is not the dataset used for this project, has not been downloaded, and is not compatible Windows telemetry. Revisit it only if access is granted and the research scope is explicitly changed.

## Current telemetry schema

`src/os_performance_prediction/telemetry.py` collects rows with `psutil` 7.0.0 and Windows system performance counters. The schema contains:

- Session ID, UTC timestamp, elapsed seconds, and optional user-supplied workload label (in the new workload collection CSV).
- CPU utilization percent; RAM utilization percent and available RAM bytes.
- System-drive capacity used percent (not disk busy time).
- Cumulative disk read/write bytes and measured bytes-per-second rates.
- Cumulative network sent/received bytes and measured bytes-per-second rates.
- Active process ID count and system-wide thread count.

Cumulative counter rate fields remain blank if counters are unavailable or reset. The collector uses monotonic time for elapsed/rate calculations, appends only to matching headers, and flushes each row. Timestamps and elapsed values can reflect small collection/scheduling overhead; consumers must align rows by actual timestamps rather than assuming perfectly regular sampling. An optional process sidecar records top CPU/memory processes at a throttled interval (60 seconds by default, off the main collection loop) using PID, process name, CPU percent, memory percent, and ranks; it never reads command-line arguments. The original Phase 3 CSV and all results/artifacts remain preserved. Full field meanings and limitations are in `docs/DATA_DICTIONARY.md`.

## Implemented research protocol and current results

### Prediction targets and features

The pipeline predicts CPU and RAM utilization 30 seconds ahead using elapsed-time interpolation within the same collection session and continuous segment. A gap over two seconds starts a new segment. Median cadence in the primary 30-minute collection was 1.0 second; two gaps exceeded two seconds, with a maximum of 48.75 seconds.

The feature builder creates 81 causal predictors: current telemetry, 1/5/10/30-second lags, and 10/30-second rolling means and standard deviations for CPU/RAM, disk capacity usage, I/O/network rates, process count, and thread count. Features use only information available at prediction time. Session and gap boundaries reset history. `available_ram_bytes` is excluded due to -0.999985 correlation with RAM utilization on this capture; cumulative counters and timestamps/session IDs are excluded. No constant feature was found in the train partition. Three high-correlation pairs (absolute Pearson correlation >= 0.995) are reported and retained for tree estimators.

### Splits and models

Chronological 70/15/15 feature-time cutoffs are used. A row is embargoed if the rightmost observation used to interpolate its future target reaches the next split boundary. The resulting 1,559 supervised rows comprise 1,060 train, 204 validation, and 234 test rows. There are 161 rows excluded for target/history/gap alignment. These exceed an explicitly operational exploratory floor (500/100/100); that floor is not a statistical guarantee. Nearly all usable observations originate in one continuous 1,620-row segment.

The experiment compares persistence with Random Forest, Extra Trees, XGBoost, LightGBM, and CatBoost independently for CPU and RAM. The test RMSEs, in utilization percentage points, are:

| Model | CPU | RAM |
|---|---:|---:|
| Persistence | 11.696 | **0.857** |
| Random Forest | 10.504 | 3.144 |
| Extra Trees | 13.338 | 2.604 |
| XGBoost | **10.153** | 1.788 |
| LightGBM | 10.428 | 2.287 |
| CatBoost | 12.400 | 0.881 |

All test MAE, MSE, RMSE, and R-squared values for train-selected models on validation and held-out test splits are in `results/windows_forecast/evaluation_metrics.csv`. CPU R-squared is negative for every test method (XGBoost -0.008, closest to zero). RAM persistence is best by RMSE and R-squared (0.137); CatBoost has lower MAE (0.572 vs 0.675) but worse RMSE/R-squared. This is weak predictive performance in this short capture; no metric is adjusted or fabricated.

Actual-vs-predicted, residual, per-target comparison, and estimator feature-importance plots are in `results/windows_forecast/`. Ten model artifacts are in `artifacts/windows_forecast/`; reloaded predictions were verified against saved test predictions. TreeExplainer explanations were generated on all 234 held-out test rows for each validation-selected target model (CPU CatBoost; RAM Random Forest). The SHAP values satisfy prediction additivity to <1e-12 maximum absolute error. Values, per-feature summaries, plots, and method/selection details are saved under `results/windows_forecast/`. SHAP describes model attribution, not causality.

## Architecture and repository modules

```text
README.md
PROJECT_PLAN.md
requirements.txt
.vscode/settings.json
data/
  raw/windows_telemetry.csv             # original Phase 3 telemetry, preserved
  raw/windows_telemetry_workload.csv     # new workload-labeled telemetry
  raw/windows_telemetry_workload_processes.csv # process name/resource sidecar
  processed/                             # engineered supervised features
src/os_performance_prediction/
  telemetry.py                            # implemented Windows collector
  validate_dataset.py                     # previous Google-trace validator; not active source
  features/windows_features.py             # causal features, targets, and splits (implemented)
  experiments/windows_forecast.py          # baselines, regressors, metrics, plots
  explainability/shap_analysis.py           # TreeExplainer outputs for held-out samples
  llm/explanations.py                     # optional API + deterministic evidence fallback
  monitoring/live_monitor.py              # live Windows psutil snapshot
  dashboard/app.py                        # six-page Streamlit application
  dashboard/data_service.py               # telemetry/model/report loading and forecast
  cli.py                                  # future application entry points
 tests/test_telemetry.py                  # implemented collector tests
artifacts/windows_forecast/              # fitted estimators
results/windows_forecast/                # actual experiment reports and plots
 docs/DATA_DICTIONARY.md                  # active telemetry schema and limitations
```

The dataset validator from the original Google plan remains in the source tree as inactive historical tooling. It is not the validator for the current Windows CSV and must not be used to imply that Google data is active.

## Application components and milestones

### Phase 0 â€” Workspace plan
Complete. Existing workspace inspected; no new top-level project structure was recreated.

### Phase 1 â€” Environment setup
Complete. Existing `.venv` is CPython 3.12.10 on Windows. Required direct dependencies are installed and import-verified in this environment; no global/Anaconda installation was performed.

### Phase 2 - Windows telemetry collection (implemented/tested)
Complete for the collector: configurable interval/duration/output, system telemetry fields, session IDs, append/header protection, and focused automated tests. The workload extension records optional user labels plus top process name/CPU/memory summaries in separate files. It was verified with short temporary live captures; no persistent workload-labeled research session has been collected yet.

### Phase 3 - Data validation, EDA, feature engineering, and split protocol
Complete. Profiled all 1,720 genuine rows and 17 columns (types, missingness, duplicates, cadence/gaps, and numeric distributions); built causal lag/rolling features, within-session 30-second targets, gap segmentation, and chronological splits with target-boundary embargo. Feature/split tests pass. See `results/windows_telemetry_profile.json` and `results/windows_forecast/data_readiness.json`.

### Phase 4 - Baselines, five regression models, and metrics
Complete for the current exploratory dataset. Persistence plus five regressors were evaluated for CPU and RAM. Validation/test metrics, predictions, fit warnings/timings, train-only feature diagnostics, ten model artifacts, and evaluation report were saved. All 19 current tests pass; model reload predictions match recorded outputs. Findings are weak and limited to this machine/session.

### Phase 5 - Plots, feature importance, and SHAP
Complete for the current experiment: genuine test comparison, actual-vs-predicted, residual, estimator feature-importance, and TreeExplainer SHAP plots. SHAP was computed on held-out rows using validation-selected models; additivity was checked.

### Phase 6 - Streamlit dashboard, workload analysis, and live Windows monitoring
Implemented. Six pages provide overview, live monitoring, historical/workload analysis, saved-model forecasts, evaluation, and SHAP/report downloads. Forecasting loads existing artifacts only (all ten are available, with caching) and uses the shared feature contract; current and historical live values are distinguished. Workload collection records optional user labels and sampled top process names/resource usage in separate files.

### Phase 7 - LLM explanation and final documentation
Implemented with a deterministic fallback and an optional OpenAI-compatible API client. The API key and optional endpoint/model are read from environment variables; API requests occur only after a user action. The fallback interprets supplied observations/predictions/SHAP and actual held-out metrics, avoids causal claims, and supplies low-risk suggestions. Live API behavior was not verified against an external provider account.

## LLM design constraints

LLM explanation is optional and separate from inference. Its input should be bounded, validated evidence (target, prediction, relevant feature contributions, uncertainty/limitations); its output should follow a small schema for explanation and suggestions. An API key must come from environment configuration. If no key/network is available, use deterministic rule-based text and label it accordingly. Do not send raw process details or unnecessary telemetry to a provider.

## Windows environment and dependencies

Target Windows 11 and 64-bit CPython 3.12.10 in `.venv`. `requirements.txt` pins the direct requested stack: pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Matplotlib, Seaborn, Plotly, Streamlit, psutil, joblib, requests, tqdm, PyArrow, Jupyter, and ipykernel. All direct pins are installed/import-verified in `.venv`; `pip check` reports no broken requirements. `ctypes` is from the Python standard library and is used for the Windows system thread count. No TensorFlow is needed for the selected tree regressors.

## Testing and quality strategy

Completed collector tests use `unittest` and live host readings in temporary files (never training fixtures). Tests cover positive interval/duration validation, counter-rate behavior on missing/reset values, live field collection, append behavior, and rejection of a mismatched CSV header. The default output was smoke-tested with genuine Windows telemetry.

Feature/target alignment, segment isolation, chronological embargo, serialization reload behavior, and SHAP additivity/alignment have been checked. Dashboard loader/page smoke tests, workload collection/summary tests, and deterministic LLM fallback tests pass. Remaining concerns include manual review on different display sizes, collecting genuine labeled workload sessions, and live-provider API verification. Full model evaluations must use validated self-collected telemetry, never synthetic records.

## Limitations and risks

- Single Windows host and limited observation windows; no claim of cross-device generalization.
- CPU/RAM/disk/network conditions depend on the user's activities during collection; labels are user supplied and may be inconsistent.
- Process enumeration is throttled (default every 60 seconds in a background worker) but still has host-dependent cost; CPU percent is interval-based and process names/PIDs may reveal installed software. No command lines or personal profile fields are collected.
- Workload-wise summaries are observational, unbalanced, and not controlled comparisons.
- Sampling cadence may drift under system load and process/device counters have OS-specific semantics.
- Disk capacity usage is not disk saturation/latency; initial telemetry has no disk busy-time measure.
- The 30-second target requires sufficient continuous session length and robust timestamp alignment.
- The current 30-minute, one-machine capture supports only an exploratory fit; its weak test results and lack of labeled workload regimes limit conclusions.
- Google Cluster Data remains an inactive access-dependent alternative and is not mixed with Windows samples.
- Explainability and LLM output can describe model associations but cannot establish causal effects.

## Current status - 2026-11-14

The Windows telemetry collector and its tests are complete. `data/raw/windows_telemetry.csv` contains 1,720 authentic rows: a five-row smoke session plus a 1,715-row, 30-minute session. Profiling found no missing values, duplicate rows, invalid timestamps, or schema deviations. The engineered dataset contains 1,559 aligned rows (1,060/204/234 chronological splits). Persistence and all five requested regressors have been evaluated; results and plots are saved under `results/windows_forecast/`, with ten fitted models under `artifacts/windows_forecast/`. All 19 collector, feature, dashboard, workload, model-loading, and fallback tests pass; artifact reload prediction checks pass. CPU test R-squared is negative across all methods; RAM persistence is best by RMSE/R-squared. Evidence is exploratory and one-machine/one-session only. As a practical next evidence collection (not a statistical guarantee), collect at least four separate 60-minute sessions on at least two days: one idle, one typical-use, one CPU-heavy, and one memory-heavy session; record the intended workload and any interruptions. Re-evaluate on chronological splits that preserve these sessions before making stronger performance claims. The six-page dashboard, live monitor, LLM fallback, and optional API client are implemented. No genuine workload-labeled research session has yet been collected; only temporary live captures have verified the new collector behavior. Live-provider API integration is unverified. Workload label/process collection uses a new CSV schema so original training/evaluation data and model artifacts remain unchanged. Google Cluster Data remains inactive.




