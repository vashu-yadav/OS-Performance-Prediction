"""Dashboard data-path and deterministic explanation tests on real saved project artifacts."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from os_performance_prediction.dashboard import data_service  # noqa: E402
from os_performance_prediction.llm.explanations import explain, rule_based_explanation  # noqa: E402


class DashboardAndExplanationTests(unittest.TestCase):
    def test_dashboard_loads_existing_telemetry_and_evaluation(self) -> None:
        telemetry = data_service.load_telemetry()
        metrics = data_service.load_metrics()
        self.assertGreater(len(telemetry), 30)
        self.assertEqual({"cpu", "ram"}, set(metrics.target))
        self.assertEqual({"test", "validation"}, set(metrics.split))

    def test_workload_comparison_summarizes_user_labeled_observations(self) -> None:
        rows = pd.DataFrame({
            "workload_label": ["Idle", "Idle", "Gaming", "Gaming"],
            "cpu_utilization_percent": [10.0, 20.0, 70.0, 80.0],
            "ram_utilization_percent": [40.0, 50.0, 60.0, 70.0],
        })
        summary, counts = data_service.summarize_workloads(rows)
        self.assertEqual(counts, {"Idle": 2, "Gaming": 2})
        self.assertEqual(summary.set_index("workload_label").loc["Gaming", "cpu_mean_percent"], 75.0)

    def test_live_forecast_uses_existing_saved_models_without_training(self) -> None:
        telemetry = data_service.load_telemetry()
        metrics = data_service.load_metrics()
        selected = data_service.validation_selected_models(metrics)
        predictions = data_service.make_forecasts(telemetry, metrics)
        self.assertEqual(set(predictions.target), {"cpu", "ram"})
        self.assertEqual(set(predictions.model), set(selected.values()))
        self.assertTrue(predictions.experimental_single_machine.all())
        self.assertTrue(predictions.predicted_percent.notna().all())

    def test_all_ten_existing_model_artifacts_load_without_fitting(self) -> None:
        registry = data_service.load_all_artifacts()
        self.assertEqual(len(registry), 10)
        self.assertEqual(set(target for target, _ in registry), {"cpu", "ram"})

    def test_llm_fallback_uses_supplied_evidence_without_api_key(self) -> None:
        evidence = {
            "current_cpu_percent": 70,
            "predicted_cpu_percent": 80,
            "current_ram_percent": 91,
            "predicted_ram_percent": 92,
            "model_quality": {"cpu": {"r2": -0.01}, "ram": {"r2": 0.13}},
            "top_shap": [{"feature": "lag_1s__cpu_utilization_percent", "mean_shap": 0.4}],
        }
        with patch.dict(os.environ, {}, clear=True):
            result = explain(evidence)
        self.assertEqual(result["source"], "rule_based")
        self.assertIn("80.0%", result["explanation"])
        self.assertIn("not causal", result["explanation"])
        self.assertTrue(result["recommendations"])

    def test_rule_fallback_reports_missing_observations_honestly(self) -> None:
        result = rule_based_explanation({})
        self.assertIn("unavailable", result["explanation"])
        self.assertIn("experimental", result["explanation"])


class DashboardPageSmokeTests(unittest.TestCase):
    def test_all_streamlit_pages_render(self) -> None:
        from streamlit.testing.v1 import AppTest

        app_path = PROJECT_ROOT / "src" / "os_performance_prediction" / "dashboard" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        self.assertFalse(app.exception, str(app.exception))
        for page in ("Live Monitoring", "Historical & Workloads", "30-Second Forecast", "Model Evaluation", "SHAP & Reports"):
            app.radio[0].set_value(page).run()
            self.assertFalse(app.exception, f"{page}: {app.exception}")


if __name__ == "__main__":
    unittest.main()
