"""Train and evaluate leakage-safe Windows CPU/RAM 30-second forecasts."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from os_performance_prediction.features.windows_features import (
    TARGETS,
    build_supervised_frame,
    chronological_split,
)
from os_performance_prediction.profile_telemetry import DEFAULT_DATA, analyze_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "windows_forecast"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "windows_forecast"
RANDOM_SEED = 42
MIN_TRAIN_ROWS = 500
MIN_VALIDATION_ROWS = 100
MIN_TEST_ROWS = 100
MODELS = ("Persistence", "Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost")


def _model_factories(seed: int) -> dict[str, Any]:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, max_features=0.8,
            random_state=seed, n_jobs=-1,
        ),
        "Extra Trees": lambda: ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=3, max_features=0.8,
            random_state=seed, n_jobs=-1,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            objective="reg:squarederror", tree_method="hist",
            random_state=seed, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=300, learning_rate=0.04, num_leaves=15,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0,
            random_state=seed, n_jobs=-1, verbosity=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=300, depth=5, learning_rate=0.04, l2_leaf_reg=3.0,
            loss_function="RMSE", random_seed=seed,
            verbose=False, allow_writing_files=False, thread_count=-1,
        ),
    }


def _metric_row(target: str, model: str, split: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "target": target,
        "model": model,
        "split": split,
        "n_rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def _validate_splits(splits: dict[str, pd.DataFrame]) -> list[str]:
    problems = []
    for name, minimum in (
        ("train", MIN_TRAIN_ROWS),
        ("validation", MIN_VALIDATION_ROWS),
        ("test", MIN_TEST_ROWS),
    ):
        count = len(splits[name])
        if count < minimum:
            problems.append(f"{name} has {count} target rows; at least {minimum} are required for this exploratory run")
    for target_name, target_column in TARGETS.items():
        for name in ("train", "validation", "test"):
            series = splits[name][f"target_{target_column}"]
            if series.nunique() < 2:
                problems.append(f"{target_name} target has fewer than two distinct values in {name}")
    return problems


def _train_constant_and_correlation_report(train: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], dict[str, Any]]:
    constant = [c for c in feature_columns if train[c].nunique(dropna=True) <= 1]
    usable = [c for c in feature_columns if c not in constant]
    corr = train[usable].corr(numeric_only=True)
    highly_correlated = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.995:
                highly_correlated.append({"feature_a": left, "feature_b": right, "pearson_r": float(value)})
    diagnostics = {
        "constant_features_dropped_using_train_only": constant,
        "high_correlation_pairs_retained_for_tree_models_threshold_0_995": highly_correlated,
        "feature_count_before_constant_drop": len(feature_columns),
        "feature_count_after_constant_drop": len(usable),
        "note": "High-correlation pairs are reported but retained; tree regressors do not require scaled orthogonal predictors, and validation/test statistics were not used for feature selection.",
    }
    return usable, diagnostics


def _write_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, importances: pd.DataFrame, results_dir: Path) -> None:
    colors = {
        "Persistence": "#666666", "Random Forest": "#1f77b4", "Extra Trees": "#ff7f0e",
        "XGBoost": "#2ca02c", "LightGBM": "#d62728", "CatBoost": "#9467bd",
    }
    targets = list(TARGETS.keys())
    metric_names = [("mae", "MAE"), ("mse", "MSE"), ("rmse", "RMSE"), ("r2", "R-squared")]
    for target in targets:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        subset = metrics[(metrics["target"] == target) & (metrics["split"] == "test")]
        x = np.arange(len(MODELS))
        for ax, (metric, label) in zip(axes.flat, metric_names):
            values = [
                subset.loc[subset.model == model, metric].iloc[0]
                for model in MODELS
            ]
            ax.bar(x, values, color=[colors[model] for model in MODELS])
            ax.set_title(f"Held-out test {label}")
            ax.set_ylabel(label)
            ax.set_xticks(x, MODELS, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{target.upper()} 30-second forecast model comparison (chronological test set)")
        fig.tight_layout()
        fig.savefig(results_dir / f"model_comparison_{target}.png", dpi=160)
        plt.close(fig)

    for target in targets:
        y_column = f"actual_{target}"
        prediction_columns = [f"pred_{target}__{name}" for name in MODELS]
        actual = predictions[y_column].to_numpy()
        low = float(min(actual.min(), predictions[prediction_columns].min().min()))
        high = float(max(actual.max(), predictions[prediction_columns].max().max()))
        fig, ax = plt.subplots(figsize=(8, 7))
        for model in MODELS:
            predicted = predictions[f"pred_{target}__{model}"].to_numpy()
            ax.scatter(actual, predicted, s=14, alpha=0.35, color=colors[model], label=model)
        ax.plot([low, high], [low, high], "k--", linewidth=1.2, label="ideal")
        ax.set_xlabel(f"Actual {target.upper()} utilization (%)")
        ax.set_ylabel(f"Predicted {target.upper()} utilization (%)")
        ax.set_title(f"Actual vs predicted — {target.upper()} (held-out test)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(results_dir / f"actual_vs_predicted_{target}.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
        for ax, model in zip(axes.flat, MODELS):
            residual = actual - predictions[f"pred_{target}__{model}"].to_numpy()
            ax.scatter(actual, residual, s=12, alpha=0.35, color=colors[model])
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(model)
            ax.set_xlabel(f"Actual {target.upper()} (%)")
            ax.set_ylabel("Residual (actual - predicted), percentage points")
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        fig.suptitle(f"Held-out residual analysis — {target.upper()}")
        fig.tight_layout()
        fig.savefig(results_dir / f"residual_analysis_{target}.png", dpi=160)
        plt.close(fig)

        target_importance = importances[importances["target"] == target]
        trained_model_names = sorted(target_importance["model"].unique())
        validation_metrics = metrics[
            (metrics.target == target)
            & (metrics.split == "validation")
            & (metrics.model.isin(trained_model_names))
        ]
        best_model = validation_metrics.sort_values("rmse").iloc[0]["model"]
        best_importance = target_importance[target_importance.model == best_model].nlargest(20, "importance")
        fig, ax = plt.subplots(figsize=(9, 7))
        if not best_importance.empty:
            plot_data = best_importance.sort_values("importance")
            ax.barh(plot_data["feature"], plot_data["importance"], color=colors[best_model])
        ax.set_title(f"Top feature importance — {target.upper()} ({best_model}, chosen on validation RMSE)")
        ax.set_xlabel("Estimator feature importance (model-specific scale)")
        fig.tight_layout()
        fig.savefig(results_dir / f"feature_importance_{target}.png", dpi=160)
        plt.close(fig)


def run_experiments(
    data_path: Path = DEFAULT_DATA,
    results_dir: Path = DEFAULT_RESULTS,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
    prepare_only: bool = False,
) -> dict[str, Any]:
    profile = analyze_dataset(data_path)
    if profile["invalid_timestamps"] or profile["duplicate_session_elapsed_rows"]:
        raise ValueError("Data failed timestamp/duplicate validation; model fitting is stopped")
    raw = pd.read_csv(data_path, low_memory=False)
    built = build_supervised_frame(raw)
    if built.frame.empty:
        raise ValueError("No 30-second target rows with complete causal history are available")
    splits, boundaries = chronological_split(built.frame)
    sufficiency_problems = _validate_splits(splits)
    data_summary = {
        "raw_rows": built.source_rows,
        "aligned_feature_target_rows": int(len(built.frame)),
        "dropped_for_alignment_or_history": built.dropped_for_alignment_or_history,
        "feature_count": len(built.feature_columns),
        "split_rows": {name: int(len(frame)) for name, frame in splits.items()},
        "split_boundaries": boundaries,
        "gap_limit_seconds": built.gap_limit_seconds,
        "session_ids_in_supervised_data": sorted(str(x) for x in built.frame["collection_session_id"].unique()),
        "exploratory_sufficiency_floor": {
            "train_rows": MIN_TRAIN_ROWS, "validation_rows": MIN_VALIDATION_ROWS, "test_rows": MIN_TEST_ROWS,
            "meaning": "Operational minimum for this within-machine exploratory model comparison, not a universal statistical guarantee.",
        },
        "sufficiency_problems": sufficiency_problems,
    }
    feature_columns, feature_diagnostics = _train_constant_and_correlation_report(splits["train"], list(built.feature_columns))
    readiness = {
        "status": "sufficient_for_exploratory_single_machine_run" if not sufficiency_problems else "insufficient",
        "data_summary": data_summary,
        "feature_diagnostics_train_only": feature_diagnostics,
        "excluded_predictors": [
            "collection_session_id, timestamps, elapsed_seconds: ordering/splitting metadata, not predictors",
            "available_ram_bytes: near-deterministic inverse of RAM utilization (full-data Pearson r reported in profile); excluded as redundant",
            "cumulative disk/network byte totals: monotonically increasing machine-history/time proxies; per-second rates used instead",
        ],
    }
    readiness_path = results_dir / "data_readiness.json"
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    processed_path = PROJECT_ROOT / "data" / "processed" / "windows_forecast_features.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    built.frame.to_parquet(processed_path, index=False)
    if prepare_only:
        print(json.dumps(readiness, indent=2))
        print(f"Saved engineered feature/target rows: {processed_path.resolve()}")
        return {**readiness, "trained": False}
    if sufficiency_problems:
        raise ValueError("Dataset does not meet the exploratory training floor: " + "; ".join(sufficiency_problems))

    if not feature_columns:
        raise ValueError("All engineered features are constant in the training partition")
    for name in ("train", "validation", "test"):
        if not np.isfinite(splits[name][feature_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite feature values remain in the {name} partition")

    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    seed = RANDOM_SEED
    factories = _model_factories(seed)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    importance_rows: list[dict[str, Any]] = []
    fit_seconds: dict[str, float] = {}
    fit_warnings: list[dict[str, str]] = []

    for target_name, target_column in TARGETS.items():
        target = f"target_{target_column}"
        train = splits["train"]
        validation = splits["validation"]
        test = splits["test"]
        y_train = train[target].to_numpy(dtype=float)
        y_validation = validation[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)
        persistence_column = f"current__{target_column}"
        prediction_frames[target_name] = pd.DataFrame({
            "collection_session_id": test["collection_session_id"].to_numpy(),
            "feature_timestamp_utc": test["feature_timestamp_utc"].astype(str).to_numpy(),
            "target_timestamp_utc": test["target_timestamp_utc"].astype(str).to_numpy(),
            f"actual_{target_name}": y_test,
        })
        persistence_predictions = {
            "validation": validation[persistence_column].to_numpy(dtype=float),
            "test": test[persistence_column].to_numpy(dtype=float),
        }
        for split_name, actual, pred in (
            ("validation", y_validation, persistence_predictions["validation"]),
            ("test", y_test, persistence_predictions["test"]),
        ):
            metric_rows.append(_metric_row(target_name, "Persistence", split_name, actual, pred))
        prediction_frames[target_name][f"pred_{target_name}__Persistence"] = persistence_predictions["test"]

        for model_name, factory in factories.items():
            estimator = factory()
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(train[feature_columns], y_train)
            fit_seconds[f"{target_name}__{model_name}"] = time.perf_counter() - start
            fit_warnings.extend({
                "target": target_name,
                "model": model_name,
                "warning": str(item.message),
            } for item in caught)
            val_prediction = np.asarray(estimator.predict(validation[feature_columns]), dtype=float)
            test_prediction = np.asarray(estimator.predict(test[feature_columns]), dtype=float)
            for split_name, actual, predicted in (
                ("validation", y_validation, val_prediction),
                ("test", y_test, test_prediction),
            ):
                metric_rows.append(_metric_row(target_name, model_name, split_name, actual, predicted))
            prediction_frames[target_name][f"pred_{target_name}__{model_name}"] = test_prediction

            importances = getattr(estimator, "feature_importances_", None)
            if importances is not None:
                importance_rows.extend({
                    "target": target_name,
                    "model": model_name,
                    "feature": feature,
                    "importance": float(value),
                } for feature, value in zip(feature_columns, importances))
            artifact_path = artifacts_dir / f"{target_name}__{model_name.lower().replace(' ', '_')}.joblib"
            joblib.dump({
                "estimator": estimator,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "horizon_seconds": 30.0,
                "random_seed": seed,
                "training_rows": len(train),
                "training_end_utc": boundaries["train_end_utc"],
            }, artifact_path)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(
        [prediction_frames[name].reset_index(drop=True) for name in TARGETS], axis=1
    )
    # The timestamp/session columns are shared; keep only their first copy.
    predictions = predictions.loc[:, ~predictions.columns.duplicated()]
    importances = pd.DataFrame(importance_rows)
    metrics.to_csv(results_dir / "evaluation_metrics.csv", index=False)
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    importances.to_csv(results_dir / "feature_importance_all_models.csv", index=False)
    (results_dir / "feature_diagnostics_train_only.json").write_text(json.dumps(feature_diagnostics, indent=2), encoding="utf-8")
    _write_plots(metrics, predictions, importances, results_dir)

    package_versions = {}
    from importlib.metadata import version
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "matplotlib", "joblib"):
        package_versions[package] = version(package)
    report = {
        "dataset": str(data_path.resolve()),
        "dataset_profile": profile,
        "data_summary": data_summary,
        "feature_diagnostics_train_only": feature_diagnostics,
        "models": list(MODELS),
        "metrics": metric_rows,
        "model_fit_seconds": fit_seconds,
        "model_fit_warnings": fit_warnings,
        "model_selection_rule_for_feature_importance": "lowest validation RMSE per target; test data was not used for this selection",
        "package_versions": package_versions,
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed": seed,
        "results_directory": str(results_dir.resolve()),
        "artifacts_directory": str(artifacts_dir.resolve()),
        "research_scope": "Exploratory chronological forecast on one Windows machine and one primary 30-minute collection. No cross-device generalization claim.",
    }
    (results_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"data_summary": data_summary, "metrics": metric_rows, "fit_seconds": fit_seconds}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--prepare-only", action="store_true", help="Profile, engineer features, and report split sufficiency without fitting models.")
    args = parser.parse_args()
    run_experiments(args.data, args.results, args.artifacts, args.prepare_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
