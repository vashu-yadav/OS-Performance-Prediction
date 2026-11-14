"""Professional Windows OS performance dashboard backed by collected telemetry."""
from __future__ import annotations

import json
from io import StringIO
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from os_performance_prediction.dashboard import data_service as service
from os_performance_prediction.llm.explanations import explain
from os_performance_prediction.monitoring.live_monitor import sample_live_metrics

st.set_page_config(page_title="Windows Performance Intelligence", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: linear-gradient(135deg,#111827,#172554); border: 1px solid #334155; padding: 14px 18px; border-radius: 14px;}
[data-testid="stSidebar"] {background: #0b1220;}
</style>
""", unsafe_allow_html=True)

st.title("Windows Performance Intelligence")
st.caption("Experimental, single-machine OS telemetry analysis and 30-second forecasting")

PAGES = ("Overview", "Live Monitoring", "Historical & Workloads", "30-Second Forecast", "Model Evaluation", "SHAP & Reports")
page = st.sidebar.radio("Application page", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Forecasts are experimental and specific to the collected machine and periods. They are not a diagnosis or cross-device guarantee.")


@st.cache_data(ttl=3)
def get_telemetry() -> pd.DataFrame:
    return service.load_telemetry()


@st.cache_data(ttl=5)
def get_processes() -> pd.DataFrame:
    return service.load_process_telemetry()


@st.cache_data(ttl=60)
def get_metrics() -> pd.DataFrame:
    return service.load_metrics()


@st.cache_data(ttl=30)
def get_predictions(data_json: str, metrics_json: str, model_selection_json: str) -> pd.DataFrame:
    data = pd.read_json(StringIO(data_json), orient="records")
    metrics = pd.read_json(StringIO(metrics_json), orient="records")
    return service.make_forecasts(data, metrics, json.loads(model_selection_json))


@st.cache_resource
def load_model_registry() -> dict:
    return service.load_all_artifacts()


def load_page_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return get_telemetry(), get_processes(), get_metrics()
    except service.DashboardDataError as exc:
        st.error(str(exc))
        st.info("Collect labeled Windows telemetry using the PowerShell command in README.md. Existing Phase 3 model artifacts and evaluation files are preserved.")
        st.stop()


def display_download(label: str, name: str, data: bytes | str, mime: str) -> None:
    st.download_button(label, data=data, file_name=name, mime=mime, use_container_width=True)


if page == "Overview":
    telemetry, _, metrics = load_page_data()
    latest = telemetry.iloc[-1]
    timestamp = latest.timestamp_utc
    st.subheader("Project overview")
    st.write("A local Windows telemetry case study with an existing, leakage-controlled 30-second CPU/RAM model evaluation. Live readings and historical predictions are deliberately shown separately.")
    cols = st.columns(4)
    for col, label, value, suffix in zip(
        cols,
        ("Latest CPU", "Latest RAM", "Disk capacity used", "Active processes"),
        (latest.cpu_utilization_percent, latest.ram_utilization_percent, latest.disk_usage_percent, latest.running_process_count),
        ("%", "%", "%", ""),
    ):
        col.metric(label, "Unavailable" if pd.isna(value) else f"{value:.1f}{suffix}")
    st.caption(f"Latest recorded sample: {timestamp.isoformat() if pd.notna(timestamp) else 'unknown'} | Sessions: {telemetry.collection_session_id.nunique()} | Rows: {len(telemetry):,}")
    cpu_status = "High" if latest.cpu_utilization_percent >= 85 else "Elevated" if latest.cpu_utilization_percent >= 70 else "Normal"
    ram_status = "High" if latest.ram_utilization_percent >= 90 else "Elevated" if latest.ram_utilization_percent >= 80 else "Normal"
    st.markdown(f"**Latest recorded system health:** CPU {cpu_status} · RAM {ram_status} · Historical record only")
    st.warning("The held-out evaluation is weak: all CPU models had negative test R², and RAM persistence outperformed the fitted regressors by RMSE. Use forecasts for demonstration, not operational decisions.")
    fig = px.line(telemetry.tail(600), x="timestamp_utc", y=["cpu_utilization_percent", "ram_utilization_percent"], labels={"value": "Utilization (%)", "timestamp_utc": "Time", "variable": "Metric"}, title="Recent historical telemetry (up to 600 rows)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Evaluation health summary")
    test = metrics[metrics.split == "test"]
    best = test.loc[test.groupby("target").rmse.idxmin(), ["target", "model", "mae", "rmse", "r2"]]
    st.dataframe(best, hide_index=True, use_container_width=True)

elif page == "Live Monitoring":
    st.subheader("Live Windows system monitoring")
    st.caption("Samples current psutil and Windows counters every two seconds. It does not append to the historical dataset. Process enumeration for collection is throttled separately.")
    if "live_history" not in st.session_state:
        st.session_state.live_history = []
    if "live_previous" not in st.session_state:
        st.session_state.live_previous = None

    @st.fragment(run_every=2)
    def live_panel() -> None:
        try:
            current = sample_live_metrics(st.session_state.live_previous)
        except (OSError, psutil.Error, PermissionError) as exc:  # noqa: F821
            st.error(f"Live Windows monitoring failed: {exc}")
            return
        st.session_state.live_previous = current
        st.session_state.live_history.append(current)
        st.session_state.live_history = st.session_state.live_history[-60:]
        cards = st.columns(4)
        cards[0].metric("CPU", f"{current['cpu_percent']:.1f}%", current["cpu_health"])
        cards[1].metric("RAM", f"{current['ram_percent']:.1f}%", current["ram_health"])
        cards[2].metric("Disk capacity", f"{current['disk_usage_percent']:.1f}%", current["disk_health"])
        cards[3].metric("Processes / threads", f"{current['process_count']:,} / {current['thread_count']:,}")
        rate_cols = st.columns(4)
        for col, label, key in zip(rate_cols, ("Network sent", "Network received", "Disk read", "Disk write"), ("network_sent_bytes_per_sec", "network_recv_bytes_per_sec", "disk_read_bytes_per_sec", "disk_write_bytes_per_sec")):
            rate = current[key]
            col.metric(label, "Warming up" if rate is None else f"{rate / 1_000_000:.2f} MB/s")
        history = pd.DataFrame(st.session_state.live_history)
        chart = px.line(history, x="timestamp_utc", y=["cpu_percent", "ram_percent"], labels={"value": "Percent", "timestamp_utc": "UTC time", "variable": "Resource"}, title="Current live CPU and RAM")
        st.plotly_chart(chart, use_container_width=True)
        st.caption("Health thresholds are heuristic UI indicators (CPU >=70/85%, RAM >=80/90%, disk >=85/95%). Disk percentage is capacity used, not disk busy time.")
        snapshot = {key: value for key, value in current.items() if key != "monotonic"}
        display_download("Download this live telemetry snapshot", "windows_live_snapshot.csv", pd.DataFrame([snapshot]).to_csv(index=False).encode("utf-8"), "text/csv")
    live_panel()

elif page == "Historical & Workloads":
    telemetry, processes, _ = load_page_data()
    st.subheader("Historical telemetry and workload context")
    if "workload_label" in telemetry and telemetry.workload_label.fillna("").astype(str).str.len().gt(0).any():
        labels = ["All"] + sorted(telemetry.workload_label.fillna("").replace("", "Unlabeled").unique().tolist())
        chosen = st.selectbox("Workload category", labels)
        view = telemetry if chosen == "All" else telemetry[telemetry.workload_label.fillna("").replace("", "Unlabeled") == chosen]
    else:
        view = telemetry
        st.info("No workload labels are present in this dataset yet. Start a new collection with --workload-label Idle, Gaming, Browsing, Development, or Multitasking. Historical unlabeled samples are preserved.")
    c1, c2 = st.columns(2)
    c1.metric("Rows in view", f"{len(view):,}")
    workload_summary, workload_counts = service.summarize_workloads(telemetry)
    c2.metric("Workload categories", str(sum(label != "Unlabeled" for label in workload_counts)))
    st.dataframe(view.tail(100), use_container_width=True, hide_index=True)
    series = [c for c in ("cpu_utilization_percent", "ram_utilization_percent") if c in view]
    if series:
        st.plotly_chart(px.line(view, x="timestamp_utc", y=series, color="collection_session_id", title="Historical resource utilization by collection session", labels={"value": "Percent", "timestamp_utc": "UTC time"}), use_container_width=True)
    io_columns = [c for c in ("disk_read_bytes_per_sec", "disk_write_bytes_per_sec", "network_sent_bytes_per_sec", "network_recv_bytes_per_sec") if c in view]
    if io_columns:
        st.plotly_chart(px.line(view.tail(1200), x="timestamp_utc", y=io_columns, title="Historical disk and network activity", labels={"value": "Bytes per second", "timestamp_utc": "UTC time", "variable": "Counter"}), use_container_width=True)
    if not workload_summary.empty:
        labeled = telemetry[telemetry.workload_label.fillna("").astype(str).str.len().gt(0)]
        st.subheader("Workload-wise resource summary")
        st.dataframe(workload_summary.round(3), hide_index=True, use_container_width=True)
        st.plotly_chart(px.box(labeled, x="workload_label", y="cpu_utilization_percent", points="outliers", title="CPU utilization by recorded workload"), use_container_width=True)
        st.plotly_chart(px.box(labeled, x="workload_label", y="ram_utilization_percent", points="outliers", title="RAM utilization by recorded workload"), use_container_width=True)
        st.caption("Comparisons are descriptive and depend on how workload labels were assigned. They do not imply controlled experiments or causation.")
    st.subheader("Top process observations")
    if processes.empty:
        st.info("No process-detail sidecar found. New collection writes process names and resource use to a separate CSV; command-line arguments are never collected.")
    else:
        latest_process_time = processes.timestamp_utc.max() if "timestamp_utc" in processes else None
        recent = processes[processes.timestamp_utc == latest_process_time] if latest_process_time is not None else processes
        st.caption(f"Latest process sample: {latest_process_time}; process names may reveal installed applications, so keep this data local.")
        st.dataframe(recent.sort_values("cpu_percent", ascending=False).head(10), hide_index=True, use_container_width=True)
        st.dataframe(recent.sort_values("memory_percent", ascending=False).head(10), hide_index=True, use_container_width=True)

elif page == "30-Second Forecast":
    telemetry, _, metrics = load_page_data()
    st.subheader("CPU and RAM forecast")
    st.warning("Experimental single-machine predictions. Models use the existing training feature contract and were not retrained in the dashboard.")
    defaults = service.validation_selected_models(metrics)
    selector_cols = st.columns(2)
    selected_models = {
        "cpu": selector_cols[0].selectbox("CPU model", service.MODEL_NAMES, index=service.MODEL_NAMES.index(defaults["cpu"])),
        "ram": selector_cols[1].selectbox("RAM model", service.MODEL_NAMES, index=service.MODEL_NAMES.index(defaults["ram"])),
    }
    st.caption("Defaults are selected by validation RMSE; choosing another entry loads its saved artifact and displays that model's held-out test metrics.")
    with st.spinner("Preparing the latest causal feature row and loading saved models..."):
        try:
            predictions = get_predictions(telemetry.to_json(orient="records", date_format="iso"), metrics.to_json(orient="records"), json.dumps(selected_models, sort_keys=True))
        except service.DashboardDataError as exc:
            predictions = None
            st.info(f"Forecast unavailable: {exc}. Collect at least 30 seconds of uninterrupted, schema-compatible telemetry first.")
    if predictions is not None:
        cols = st.columns(2)
        for col, (_, row) in zip(cols, predictions.iterrows()):
            col.metric(f"30-second {row.target.upper()} prediction", f"{row.predicted_percent:.1f}%")
            col.write(f"Model: **{row.model}** · feature time: {row.feature_timestamp_utc}")
            col.caption(f"Historical held-out test: MAE {row.test_mae:.3f}, MSE {row.test_mse:.3f}, RMSE {row.test_rmse:.3f}, R² {row.test_r2:.3f}")
        st.dataframe(predictions, hide_index=True, use_container_width=True)
        display_download("Download forecast results", "windows_30_second_predictions.csv", predictions.to_csv(index=False).encode("utf-8"), "text/csv")
        st.caption("Current-feature construction uses the newest uninterrupted 30+ second history. If collection ends long before this dashboard is opened, the computed forecast timestamp refers to the latest sample time, not wall-clock now.")
        st.subheader("Evidence-based explanation")
        try:
            shap_summary = pd.read_csv(service.RESULTS / "shap_feature_summary.csv")
            top_shap = shap_summary.sort_values("mean_abs_shap", ascending=False).groupby("target").head(5).to_dict("records")
            st.caption("SHAP evidence below summarizes the held-out evaluation rows; it is not a row-specific explanation of this new forecast.")
        except (OSError, pd.errors.ParserError):
            top_shap = []
        latest = telemetry.iloc[-1]
        evidence = {
            "current_cpu_percent": float(latest.cpu_utilization_percent),
            "current_ram_percent": float(latest.ram_utilization_percent),
            "predicted_cpu_percent": float(predictions.loc[predictions.target == "cpu", "predicted_percent"].iloc[0]),
            "predicted_ram_percent": float(predictions.loc[predictions.target == "ram", "predicted_percent"].iloc[0]),
            "model_quality": {row.target: {"r2": row.test_r2, "rmse": row.test_rmse} for _, row in predictions.iterrows()},
            "top_shap": top_shap,
            "scope": "one Windows machine, exploratory historical data",
        }
        if st.button("Explain these predictions", type="primary"):
            with st.spinner("Preparing evidence-grounded explanation..."):
                result = explain(evidence)
            st.info(f"Explanation source: {result['source']}")
            st.write(result["explanation"])
            for recommendation in result["recommendations"]:
                st.markdown(f"- {recommendation}")
            if result.get("api_error"):
                st.caption(f"Optional API was unavailable; deterministic fallback used ({result['api_error']}).")

elif page == "Model Evaluation":
    telemetry, _, metrics = load_page_data()
    st.subheader("Existing Phase 3 model evaluation")
    st.caption("No training is performed here. Values below are from saved validation and chronological held-out test results.")
    if st.button("Load and validate all ten saved model artifacts"):
        with st.spinner("Loading the existing model artifacts (no fitting)..."):
            try:
                registry = load_model_registry()
                st.success(f"Loaded {len(registry)} saved CPU/RAM model artifacts.")
            except service.DashboardDataError as exc:
                st.error(str(exc))
    target_filter = st.selectbox("Target", ["Both", "cpu", "ram"])
    split_filter = st.selectbox("Evaluation partition", ["test", "validation", "Both"])
    selected = metrics.copy()
    if target_filter != "Both": selected = selected[selected.target == target_filter]
    if split_filter != "Both": selected = selected[selected.split == split_filter]
    st.dataframe(selected.sort_values(["target", "split", "rmse"]), hide_index=True, use_container_width=True)
    metric_choice = st.selectbox("Comparison metric", ["MAE", "MSE", "RMSE", "R²"])
    metric_column = {"MAE": "mae", "MSE": "mse", "RMSE": "rmse", "R²": "r2"}[metric_choice]
    metric_label = "R-squared" if metric_column == "r2" else f"{metric_choice} (percentage-point scale)"
    chart_data = selected.copy()
    st.plotly_chart(px.bar(chart_data, x="model", y=metric_column, color="target", facet_col="split" if split_filter == "Both" else None, barmode="group", title=f"Model {metric_choice} comparison", labels={metric_column: metric_label}), use_container_width=True)
    col1, col2 = st.columns(2)
    for col, filename, title in ((col1, "actual_vs_predicted_cpu.png", "CPU actual vs predicted"), (col2, "actual_vs_predicted_ram.png", "RAM actual vs predicted")):
        path = service.RESULTS / filename
        if path.is_file(): col.image(str(path), caption=title, use_container_width=True)
    col1, col2 = st.columns(2)
    for col, filename, title in ((col1, "residual_analysis_cpu.png", "CPU residuals"), (col2, "residual_analysis_ram.png", "RAM residuals")):
        path = service.RESULTS / filename
        if path.is_file(): col.image(str(path), caption=title, use_container_width=True)
    display_download("Download all evaluation metrics", "evaluation_metrics.csv", metrics.to_csv(index=False).encode("utf-8"), "text/csv")

else:  # SHAP & Reports
    telemetry, _, metrics = load_page_data()
    st.subheader("SHAP explanations and downloadable reports")
    shap_path = service.RESULTS / "shap_feature_summary.csv"
    if shap_path.is_file():
        shap_summary = pd.read_csv(shap_path)
        target = st.selectbox("SHAP target", ["cpu", "ram"])
        shap_view = shap_summary[shap_summary.target == target].sort_values("mean_abs_shap", ascending=False).head(20)
        st.plotly_chart(px.bar(shap_view.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", title=f"Top SHAP features: {target.upper()}", labels={"mean_abs_shap": "Mean absolute SHAP value"}), use_container_width=True)
        path = service.RESULTS / f"shap_summary_{target}.png"
        if path.is_file(): st.image(str(path), caption=f"SHAP beeswarm - {target.upper()}", use_container_width=True)
        st.dataframe(shap_view, hide_index=True, use_container_width=True)
        display_download("Download SHAP feature summary", "shap_feature_summary.csv", shap_summary.to_csv(index=False).encode("utf-8"), "text/csv")
        st.caption("SHAP describes how the fitted model distributes output across observed features. It does not establish cause or justify an intervention.")
    else:
        st.warning("Saved SHAP summary is unavailable. No replacement explanation is fabricated.")
    st.subheader("Dataset and collection statistics")
    st.json({"telemetry_rows": len(telemetry), "columns": list(telemetry.columns), "collection_sessions": int(telemetry.collection_session_id.nunique()), "first_sample_utc": telemetry.timestamp_utc.min().isoformat(), "last_sample_utc": telemetry.timestamp_utc.max().isoformat(), "missing_values": telemetry.isna().sum().to_dict(), "workload_counts": service.summarize_workloads(telemetry)[1]})
    display_download("Download collected telemetry dataset", "windows_telemetry_snapshot.csv", telemetry.to_csv(index=False).encode("utf-8"), "text/csv")
    display_download("Download model comparison metrics", "evaluation_metrics.csv", metrics.to_csv(index=False).encode("utf-8"), "text/csv")
    try:
        defaults = service.validation_selected_models(metrics)
        forecast_bytes = get_predictions(telemetry.to_json(orient="records", date_format="iso"), metrics.to_json(orient="records"), json.dumps(defaults, sort_keys=True)).to_csv(index=False).encode("utf-8")
    except service.DashboardDataError:
        forecast_bytes = None
    if forecast_bytes is not None:
        display_download("Download prediction results", "windows_30_second_predictions.csv", forecast_bytes, "text/csv")
    else:
        st.info("Prediction download unavailable: insufficient uninterrupted history or a missing model artifact. No placeholder result was generated.")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "telemetry_file": str(service.available_telemetry_path()),
        "row_count": int(len(telemetry)),
        "session_count": int(telemetry.collection_session_id.nunique()),
        "workload_counts": service.summarize_workloads(telemetry)[1],
        "latest_cpu_percent": float(telemetry.iloc[-1].cpu_utilization_percent),
        "latest_ram_percent": float(telemetry.iloc[-1].ram_utilization_percent),
        "test_results": metrics[metrics.split == "test"].to_dict("records"),
        "limitations": ["Single-machine telemetry", "Short collection with weak forecast results", "Workload labels are user assigned", "SHAP is not causal"],
    }
    display_download("Download system analysis report", "windows_system_analysis.json", json.dumps(report, indent=2).encode("utf-8"), "application/json")
