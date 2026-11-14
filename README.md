# AI-Driven Windows OS Performance Prediction and Intelligent Resource Analysis

B.Tech CSE major project, UPES.

## Current research dataset

The active dataset strategy is an **authentic, self-collected Windows telemetry time series** from this machine using `psutil`. The Phase 3 model/evaluation source is `data/raw/windows_telemetry.csv`; the enhanced collector writes new labeled sessions to the separate workload-aware files documented below. This is a single-machine dataset; it describes this computer and collection period only. It does not establish that models generalize to other Windows computers, hardware, users, or workloads.

The Google Cluster Data 2011-2 option is documented as a pending/abandoned alternative because official access approval was taking too long. It is not the dataset used by this project. No Google trace data has been downloaded, and no model is trained from it.

## Environment

Use the existing Windows virtual environment and Python 3.12.10:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip check
```

All dependencies are pinned in `requirements.txt` and were installed and import-verified in `.venv` on Windows.

## Collect labeled Windows telemetry

From the project root in PowerShell, choose the label that best describes the session. Labels are optional and user-assigned.

```powershell
.\.venv\Scripts\python.exe src\os_performance_prediction\telemetry.py --interval-seconds 1 --duration-seconds 1800 --workload-label Development
```

The enhanced collector writes aggregate rows to `data/raw/windows_telemetry_workload.csv` and sampled process details to `data/raw/windows_telemetry_workload_processes.csv`. It defaults to enumerating the top five CPU and memory processes every 60 seconds in a background worker to limit overhead; adjust with `--top-processes` and `--process-sampling-interval-seconds`. Available labels: `Idle`, `Gaming`, `Browsing`, `Development`, and `Multitasking`. To omit a label, leave out `--workload-label`.

The earlier `data/raw/windows_telemetry.csv` dataset remains unchanged and remains the source for the saved Phase 3 model results. New collection uses a versioned file because the aggregate schema now includes `workload_label`; the existing model feature builder ignores this context field and uses the same trained 81-feature contract. It does not retrain models. Existing incompatible CSV headers are rejected rather than modified.

A one-time local timing check on this host (336 visible PIDs) measured a process snapshot at about 0.95 seconds; it executes off the main telemetry loop and defaults to once per minute. Host cost varies. Process details contain PID, process name, CPU percentage, memory percentage, and top-N ranks only. Command-line arguments, user names, executable paths, and window titles are not collected. Process names can still reveal installed applications; keep these files local. Process CPU percentage is sampled over the configured process interval and can exceed 100% on multicore systems. Disk utilization is system-drive capacity used, not disk busy time.

## Phase 3 ML pipeline and evaluation

The raw CSV has 1,720 genuine rows across a five-row smoke session and a 30-minute session. Profiling found 17 expected columns, no missing values, duplicates, or invalid timestamps. Main-session median cadence is one second, with two gaps over two seconds (maximum 48.75 seconds); feature history and targets reset at these gaps. Detailed distributions are in `results/windows_telemetry_profile.json`.

The feature pipeline creates current values, causal 1/5/10/30-second lags, and 10/30-second rolling means and standard deviations. CPU and RAM targets are aligned 30 seconds ahead within continuous segments. Chronological 70/15/15 split boundaries embargo labels whose interpolation bracket crosses the next boundary. The result is 1,559 supervised rows (1,060 train, 204 validation, 234 test). This supports an exploratory run only and does not provide labeled workload-regime or cross-device coverage.

Held-out chronological test RMSE (utilization percentage points):

| Model | CPU RMSE | RAM RMSE |
|---|---:|---:|
| Persistence | 11.696 | **0.857** |
| Random Forest | 10.504 | 3.144 |
| Extra Trees | 13.338 | 2.604 |
| XGBoost | **10.153** | 1.788 |
| LightGBM | 10.428 | 2.287 |
| CatBoost | 12.400 | 0.881 |

Every CPU model has negative test R-squared (XGBoost is closest to zero at -0.008). RAM persistence has the best RMSE and R-squared (0.137); CatBoost has lower MAE but worse RMSE and R-squared. Results indicate weak predictive performance in this capture. All validation/test MAE, MSE, RMSE, and R-squared values are in `results/windows_forecast/evaluation_metrics.csv`; the detailed report, test predictions, plots, diagnostics, and ten saved artifacts are also available in the repository.

Implemented and tested: telemetry collection, full CSV profiling, causal feature/target engineering, leakage-controlled splits, persistence baseline, five regressors, evaluation, comparison/actual-vs-predicted/residual/feature-importance figures. The new dashboard, workload collector, and LLM fallback are also covered by tests. All 19 current tests pass; existing models reproduce the saved forecasts.

SHAP TreeExplainer outputs are generated for all 234 held-out rows using validation-selected models (CPU CatBoost, RAM Random Forest). SHAP additivity was verified to below 1e-12 maximum absolute error. See `results/windows_forecast/shap_report.json`, `shap_values_test.csv`, `shap_feature_summary.csv`, and the two summary plots. SHAP explains the fitted model output; it does not establish causality. The Streamlit dashboard and LLM/rule-based fallback are implemented. No genuine workload-labeled research session has yet been collected; the collector feature was verified with short temporary host captures only. As a practical next data collection, gather at least four separate 60-minute sessions across at least two days (idle, typical use, CPU-heavy, memory-heavy), recording workload context; this is a collection plan, not a statistical guarantee.


## Streamlit application

Launch the local application from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\os_performance_prediction\dashboard\app.py
```

The six pages are Overview, Live Monitoring, Historical & Workloads, 30-Second Forecast, Model Evaluation, and SHAP & Reports. The forecast page loads the existing ten model artifacts and defaults to each target's saved validation-RMSE model and lets users select any of the five saved regressors. All ten artifacts are loadable from the evaluation page. It requires at least 30 seconds of uninterrupted telemetry history and displays the model's actual held-out test metrics. It never trains or fabricates a forecast. The optional LLM explanation uses `OS_PERFORMANCE_LLM_API_KEY` from the environment and optionally `OS_PERFORMANCE_LLM_BASE_URL` / `OS_PERFORMANCE_LLM_MODEL`. Without a key it uses the deterministic evidence-based fallback. API calls are user-triggered; process details are not sent. The API request path is implemented but has not been verified against a live provider account.

For Streamlit Community Cloud, set the app's main file to `src/os_performance_prediction/dashboard/app.py`. Do not use `src/os_performance_prediction/explainability/shap_analysis.py` as the app file; it is an explicit offline analysis utility. Raw telemetry is intentionally not committed. Without a local CSV, the app still displays saved model evaluation, artifacts, and SHAP summaries; historical and forecast pages explain when local collection is needed.

Workload summaries are descriptive comparisons of user-labeled sessions, not controlled experiments. Do not use the forecasts for safety-critical or automatic resource management decisions.

## Privacy and limitations

The CSV includes host-level resource activity and cumulative network/disk counters, but the new optional process sidecar includes process names and PIDs (never command lines). Keep collected files local unless sharing is intended. Process/thread counts can omit inaccessible or short-lived processes. Sampling time can drift under load. Model results from one computer must be presented as a single-machine case study, not a general Windows performance claim.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for milestones and [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for the telemetry schema and dataset status.



