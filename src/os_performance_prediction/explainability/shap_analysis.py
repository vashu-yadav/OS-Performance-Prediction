"""Create SHAP explanations for validation-selected models on held-out telemetry."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "windows_telemetry.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "windows_forecast"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "windows_forecast"
MODEL_NAMES = ("Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost")


def create_shap_outputs(
    data_path: Path = DEFAULT_DATA,
    results_dir: Path = DEFAULT_RESULTS,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
) -> dict[str, object]:
    # Heavy analysis dependencies and all data access stay inside this explicit
    # call so importing this module is safe in Streamlit and other applications.
    import joblib
    import matplotlib
    import numpy as np
    import pandas as pd
    import shap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from os_performance_prediction.features.windows_features import (  # noqa: PLC0415
        build_supervised_frame,
        chronological_split,
    )

    raw = pd.read_csv(data_path, low_memory=False)
    built = build_supervised_frame(raw)
    splits, _ = chronological_split(built.frame)
    metrics = pd.read_csv(results_dir / "evaluation_metrics.csv")
    test_predictions = pd.read_csv(results_dir / "test_predictions.csv")
    records: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    selections: dict[str, str] = {}

    for target, target_column in (("cpu", "cpu_utilization_percent"), ("ram", "ram_utilization_percent")):
        validation = metrics[(metrics.target == target) & (metrics.split == "validation") & metrics.model.isin(MODEL_NAMES)]
        model_name = str(validation.sort_values("rmse").iloc[0].model)
        selections[target] = model_name
        artifact_name = model_name.lower().replace(" ", "_")
        artifact = joblib.load(artifacts_dir / f"{target}__{artifact_name}.joblib")
        estimator = artifact["estimator"]
        feature_columns = artifact["feature_columns"]
        X_test = splits["test"][feature_columns].reset_index(drop=True)
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(X_test)
        values = np.asarray(explanation.values)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.shape != X_test.shape:
            raise ValueError(f"Unexpected SHAP shape for {target}: {values.shape}; features: {X_test.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite SHAP values found for {target}")

        frame = pd.DataFrame(values, columns=[f"shap_{name}" for name in feature_columns])
        frame.insert(0, "target", target)
        frame.insert(1, "model", model_name)
        frame.insert(2, "feature_timestamp_utc", test_predictions["feature_timestamp_utc"].astype(str).to_numpy())
        frame.insert(3, "actual_target_percent", test_predictions[f"actual_{target}"].to_numpy(dtype=float))
        frame.insert(4, "predicted_target_percent", test_predictions[f"pred_{target}__{model_name}"].to_numpy(dtype=float))
        frame.insert(5, "expected_value", np.asarray(explanation.base_values, dtype=float).reshape(-1))
        records.append(frame)

        summary = pd.DataFrame({
            "target": target,
            "model": model_name,
            "feature": feature_columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        summaries.append(summary)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(explanation, X_test, max_display=20, show=False)
        plt.title(f"{target.upper()} held-out SHAP summary ({model_name}, selected on validation RMSE)")
        plt.tight_layout()
        plt.savefig(results_dir / f"shap_summary_{target}.png", dpi=160, bbox_inches="tight")
        plt.close()

    combined = pd.concat(records, ignore_index=True)
    combined.to_csv(results_dir / "shap_values_test.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(results_dir / "shap_feature_summary.csv", index=False)
    report = {
        "method": "TreeExplainer",
        "selection": "lowest validation RMSE among five fitted regressors, independently per target",
        "explained_partition": "chronological held-out test rows; never used for model selection",
        "test_rows_per_target": int(len(splits["test"])),
        "feature_count": len(built.feature_columns),
        "selected_models": selections,
        "shap_version": shap.__version__,
        "outputs": ["shap_values_test.csv", "shap_feature_summary.csv", "shap_summary_cpu.png", "shap_summary_ram.png"],
        "interpretation_limit": "SHAP attributes model output across observed features; it does not establish causal effects or cross-device generalization.",
    }
    (results_dir / "shap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _is_running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def main(argv: list[str] | None = None, streamlit_context: bool | None = None) -> int:
    if streamlit_context is None:
        streamlit_context = _is_running_inside_streamlit()
    if streamlit_context:
        import streamlit as st

        st.error(
            "This is the offline SHAP generation utility, not the dashboard entry point. "
            "Configure Streamlit Community Cloud to run "
            "src/os_performance_prediction/dashboard/app.py. "
            "SHAP generation is not run during dashboard startup."
        )
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Explicitly run SHAP generation from local telemetry and saved test predictions.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)
    if not args.generate:
        print("No SHAP work was run. To generate SHAP outputs explicitly, pass --generate.")
        return 0
    print(json.dumps(create_shap_outputs(args.data, args.results, args.artifacts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
