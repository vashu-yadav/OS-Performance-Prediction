"""Dashboard data-path and deterministic explanation tests on real saved project artifacts."""
from __future__ import annotations

import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
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

    def test_dashboard_starts_without_raw_telemetry_and_keeps_saved_results_available(self) -> None:
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        st.cache_data.clear()
        missing = PROJECT_ROOT / "data" / "raw" / "deployment_test_data_not_present.csv"
        app_path = PROJECT_ROOT / "src" / "os_performance_prediction" / "dashboard" / "app.py"
        with patch.object(data_service, "RAW_PRIMARY", missing), patch.object(data_service, "RAW_LEGACY", missing):
            app = AppTest.from_file(str(app_path), default_timeout=60).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any("Historical telemetry is not present" in item.value for item in app.info))
            for page in ("Historical & Workloads", "30-Second Forecast", "Model Evaluation", "SHAP & Reports"):
                app.radio[0].set_value(page).run()
                self.assertFalse(app.exception, f"{page} without raw CSV: {app.exception}")
            self.assertGreater(len(app.get("plotly_chart")), 0)
            self.assertGreater(len(app.get("json")), 0)

    def test_missing_raw_csv_is_reported_as_optional_dashboard_data(self) -> None:
        import tempfile

        missing = Path(tempfile.gettempdir()) / "os_performance_missing_telemetry_regression.csv"
        self.assertFalse(missing.exists())
        with self.assertRaisesRegex(data_service.DashboardDataError, "Telemetry file not found"):
            data_service.load_telemetry(missing)

    def test_offline_data_pipelines_refuse_to_run_as_streamlit_entrypoints(self) -> None:
        from unittest.mock import Mock
        from os_performance_prediction import profile_telemetry, runtime_guard
        from os_performance_prediction.experiments import windows_forecast

        with patch.object(runtime_guard, "is_streamlit_context", return_value=True):
            with patch("streamlit.error", new=Mock()) as streamlit_error:
                self.assertEqual(profile_telemetry.main(), 0)
                self.assertEqual(windows_forecast.main(), 0)
                self.assertEqual(streamlit_error.call_count, 2)

    def test_shap_utility_does_not_run_pipeline_inside_streamlit(self) -> None:
        from unittest.mock import Mock
        from os_performance_prediction.explainability import shap_analysis

        with patch.object(shap_analysis, "create_shap_outputs", side_effect=AssertionError("pipeline must not run")):
            with patch("streamlit.error", new=Mock()) as streamlit_error:
                self.assertEqual(shap_analysis.main(["--generate"], streamlit_context=True), 0)
                streamlit_error.assert_called_once()

    def test_importing_or_running_shap_utility_without_generate_does_not_read_data(self) -> None:
        from os_performance_prediction.explainability import shap_analysis

        with patch.object(shap_analysis, "create_shap_outputs", side_effect=AssertionError("pipeline must not run")):
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(shap_analysis.main([], streamlit_context=False), 0)
            self.assertIn("pass --generate", output.getvalue())


if __name__ == "__main__":
    unittest.main()
