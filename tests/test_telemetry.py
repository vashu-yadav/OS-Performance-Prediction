"""Tests for Windows telemetry collection; captured rows are ephemeral test data."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from os_performance_prediction import telemetry  # noqa: E402


class TelemetryTests(unittest.TestCase):
    def test_counter_rates_preserve_missing_and_counter_resets(self) -> None:
        self.assertEqual(telemetry._rate(100, 160, 2.0), 30.0)
        self.assertIsNone(telemetry._rate(None, 160, 2.0))
        self.assertIsNone(telemetry._rate(160, 100, 2.0))
        self.assertIsNone(telemetry._rate(100, 160, 0.0))

    def test_cli_rejects_nonpositive_collection_values(self) -> None:
        with self.assertRaises(SystemExit) as error:
            telemetry.parse_args(["--interval-seconds", "0"])
        self.assertEqual(error.exception.code, 2)
        with self.assertRaises(SystemExit) as error:
            telemetry.parse_args(["--duration-seconds", "-1"])
        self.assertEqual(error.exception.code, 2)

    def test_live_windows_collection_writes_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "telemetry_test.csv"
            samples = telemetry.collect_telemetry(
                output_path=output,
                interval_seconds=0.05,
                duration_seconds=0.35,
            )
            self.assertGreaterEqual(samples, 1)
            with output.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), telemetry.COLUMNS)
            self.assertEqual(len(rows), samples)
            self.assertTrue(rows[0]["collection_session_id"])
            self.assertTrue(rows[0]["timestamp_utc"].endswith("Z"))
            for field in (
                "cpu_utilization_percent",
                "ram_utilization_percent",
                "available_ram_bytes",
                "disk_usage_percent",
                "running_process_count",
                "thread_count",
            ):
                self.assertNotEqual(rows[0][field], "")

    def test_collection_appends_to_a_matching_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "telemetry_test.csv"
            first_count = telemetry.collect_telemetry(output, 0.04, 0.08)
            second_count = telemetry.collect_telemetry(output, 0.04, 0.08)
            with output.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), first_count + second_count)
            self.assertGreaterEqual(first_count, 1)
            self.assertGreaterEqual(second_count, 1)
            self.assertEqual(len({row["collection_session_id"] for row in rows}), 2)

    def test_process_and_thread_counts_are_available_on_windows(self) -> None:
        process_count, thread_count = telemetry._process_counts()
        self.assertGreater(process_count, 0)
        self.assertGreater(thread_count, 0)

    def test_existing_csv_with_wrong_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "telemetry_test.csv"
            output.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                telemetry._open_writer(output)

    def test_workload_label_and_top_process_sidecar_are_written_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "workload.csv"
            process_output = Path(directory) / "processes.csv"
            telemetry.collect_telemetry(
                output, interval_seconds=0.05, duration_seconds=0.50,
                workload_label="Development", top_processes=3,
                process_sampling_interval_seconds=0.05, process_output_path=process_output,
            )
            with output.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with process_output.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                process_rows = list(reader)
                process_columns = tuple(reader.fieldnames or ())
            self.assertTrue(rows)
            self.assertTrue(all(row["workload_label"] == "Development" for row in rows))
            self.assertTrue(process_rows)
            self.assertTrue(all(row["workload_label"] == "Development" for row in process_rows))
            self.assertIn("process_name", process_columns)
            self.assertIn("cpu_percent", process_columns)
            self.assertIn("memory_percent", process_columns)
            self.assertFalse(any("cmdline" in column.lower() or "command" in column.lower() for column in process_columns))


if __name__ == "__main__":
    unittest.main()
