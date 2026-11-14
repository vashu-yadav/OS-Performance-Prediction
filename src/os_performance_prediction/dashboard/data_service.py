"""Read validated project data and existing models for dashboard views."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PRIMARY = PROJECT_ROOT / "data" / "raw" / "windows_telemetry_workload.csv"
RAW_LEGACY = PROJECT_ROOT / "data" / "raw" / "windows_telemetry.csv"
PROCESS_DATA = PROJECT_ROOT / "data" / "raw" / "windows_telemetry_workload_processes.csv"
RESULTS = PROJECT_ROOT / "results" / "windows_forecast"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "windows_forecast"
MODEL_NAMES = ("Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost")


class DashboardDataError(RuntimeError):
    """A dashboard input is absent, malformed, or incompatible."""


def available_telemetry_path() -> Path:
    for path in (RAW_PRIMARY, RAW_LEGACY):
        if path.is_file() and path.stat().st_size:
            return path
    raise DashboardDataError("No telemetry CSV found. Start a labeled collection from the README instructions.")


def load_telemetry(path: Path | None = None) -> pd.DataFrame:
    source = path or available_telemetry_path()
    if not source.is_file():
        raise DashboardDataError(f"Telemetry file not found: {source}")
    try:
        frame = pd.read_csv(source, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(f"Could not read telemetry CSV: {exc}") from exc
    required = {
        "collection_session_id", "timestamp_utc", "elapsed_seconds", "cpu_utilization_percent",
        "ram_utilization_percent", "disk_usage_percent", "running_process_count", "thread_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DashboardDataError(f"Telemetry schema is missing required fields: {', '.join(missing)}")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    if frame.empty:
        raise DashboardDataError("Telemetry has no valid timestamped rows")
    return frame


def load_process_telemetry(path: Path = PROCESS_DATA) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["timestamp_utc", "workload_label", "pid", "process_name", "cpu_percent", "memory_percent", "cpu_rank", "memory_rank"])
    try:
        frame = pd.read_csv(path, low_memory=False)
        if "timestamp_utc" in frame:
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        return frame
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(f"Could not read process telemetry: {exc}") from exc


def load_metrics() -> pd.DataFrame:
    path = RESULTS / "evaluation_metrics.csv"
    if not path.is_file():
        raise DashboardDataError("Existing evaluation metrics are unavailable; run the Phase 3 experiment first.")
    frame = pd.read_csv(path)
    expected = {"target", "model", "split", "n_rows", "mae", "mse", "rmse", "r2"}
    if not expected.issubset(frame.columns):
        raise DashboardDataError("Evaluation metrics file has an incompatible schema")
    return frame


def summarize_workloads(telemetry: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return descriptive CPU/RAM summaries and counts by user-supplied label."""
    labels = telemetry.get("workload_label", pd.Series("Unlabeled", index=telemetry.index))
    labels = labels.fillna("").astype(str).replace("", "Unlabeled")
    frame = telemetry.assign(_workload=labels)
    counts = {str(label): int(count) for label, count in frame._workload.value_counts().items()}
    if set(counts) == {"Unlabeled"}:
        return pd.DataFrame(), counts
    summary = frame.groupby("_workload").agg(
        samples=("cpu_utilization_percent", "size"),
        cpu_mean_percent=("cpu_utilization_percent", "mean"),
        cpu_median_percent=("cpu_utilization_percent", "median"),
        ram_mean_percent=("ram_utilization_percent", "mean"),
        ram_median_percent=("ram_utilization_percent", "median"),
    ).reset_index().rename(columns={"_workload": "workload_label"})
    return summary, counts


@lru_cache(maxsize=10)
def load_artifact(target: str, model: str) -> dict[str, Any]:
    filename = model.lower().replace(" ", "_")
    path = ARTIFACTS / f"{target}__{filename}.joblib"
    if not path.is_file():
        raise DashboardDataError(f"Saved {target.upper()} model artifact is missing: {path.name}")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise DashboardDataError(f"Could not load saved model {path.name}: {exc}") from exc
    if "estimator" not in artifact or "feature_columns" not in artifact:
        raise DashboardDataError(f"Saved model {path.name} is missing its feature contract")
    return artifact


def load_all_artifacts() -> dict[tuple[str, str], dict[str, Any]]:
    """Load and cache every existing target/model artifact without fitting."""
    return {
        (target, model): load_artifact(target, model)
        for target in ("cpu", "ram")
        for model in MODEL_NAMES
    }


def validation_selected_models(metrics: pd.DataFrame | None = None) -> dict[str, str]:
    metrics = metrics if metrics is not None else load_metrics()
    selected: dict[str, str] = {}
    for target in ("cpu", "ram"):
        candidates = metrics[(metrics.target == target) & (metrics.split == "validation") & metrics.model.isin(MODEL_NAMES)]
        if candidates.empty:
            raise DashboardDataError(f"No validation results are available for {target.upper()}")
        selected[target] = str(candidates.sort_values("rmse").iloc[0].model)
    return selected


def make_forecasts(
    telemetry: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    selected_models: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Predict from the newest causal history; model artifacts are loaded, never fitted."""
    metrics = metrics if metrics is not None else load_metrics()
    selected = validation_selected_models(metrics)
    if selected_models:
        for target, model in selected_models.items():
            if target not in selected or model not in MODEL_NAMES:
                raise DashboardDataError(f"Unsupported saved model selection: {target} / {model}")
            selected[target] = model
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from os_performance_prediction.features.windows_features import build_latest_inference_row  # noqa: PLC0415

    try:
        feature_row, metadata = build_latest_inference_row(telemetry)
    except (ValueError, KeyError) as exc:
        raise DashboardDataError(str(exc)) from exc
    output: list[dict[str, Any]] = []
    for target, target_col in (("cpu", "cpu_utilization_percent"), ("ram", "ram_utilization_percent")):
        model = selected[target]
        artifact = load_artifact(target, model)
        columns = artifact["feature_columns"]
        missing = sorted(set(columns) - set(feature_row.columns))
        if missing:
            raise DashboardDataError(f"Latest telemetry lacks trained features: {', '.join(missing)}")
        prediction = float(artifact["estimator"].predict(feature_row[columns])[0])
        if not np.isfinite(prediction):
            raise DashboardDataError(f"{target.upper()} model returned a non-finite prediction")
        test_metric = metrics[(metrics.target == target) & (metrics.model == model) & (metrics.split == "test")]
        if test_metric.empty:
            raise DashboardDataError(f"Held-out test performance is missing for {target.upper()} / {model}")
        score = test_metric.iloc[0]
        output.append({
            "target": target,
            "predicted_percent": prediction,
            "model": model,
            "feature_timestamp_utc": metadata["feature_timestamp_utc"],
            "forecast_timestamp_utc": (pd.Timestamp(metadata["feature_timestamp_utc"]) + pd.Timedelta(seconds=30)).isoformat(),
            "test_mae": float(score.mae), "test_mse": float(score.mse),
            "test_rmse": float(score.rmse), "test_r2": float(score.r2),
            "experimental_single_machine": True,
        })
    return pd.DataFrame(output)


def load_json_report(filename: str) -> dict[str, Any]:
    path = RESULTS / filename
    if not path.is_file():
        raise DashboardDataError(f"Report is missing: {filename}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Could not read report {filename}: {exc}") from exc
