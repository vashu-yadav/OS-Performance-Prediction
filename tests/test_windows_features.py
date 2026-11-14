"""Unit tests use small synthetic frames only as isolated code fixtures, never for training."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from os_performance_prediction.features.windows_features import (  # noqa: E402
    SOURCE_FEATURES,
    _interpolate_at,
    build_latest_inference_row,
    build_supervised_frame,
    chronological_split,
)


def make_fixture(session_id: str, start_time: str, count: int = 180) -> pd.DataFrame:
    times = np.arange(1, count + 1, dtype=float)
    utc_start = pd.Timestamp(start_time, tz="UTC")
    frame = pd.DataFrame({
        "collection_session_id": session_id,
        "timestamp_utc": [(utc_start + pd.Timedelta(seconds=float(t))).isoformat() for t in times],
        "elapsed_seconds": times,
        "cpu_utilization_percent": 30 + 10 * np.sin(times / 15),
        "ram_utilization_percent": 70 + 2 * np.sin(times / 50),
        "disk_usage_percent": 60 + (times % 3) * 0.1,
        "disk_read_bytes_per_sec": 1000 + (times % 7),
        "disk_write_bytes_per_sec": 2000 + (times % 5),
        "network_sent_bytes_per_sec": 300 + (times % 9),
        "network_recv_bytes_per_sec": 400 + (times % 11),
        "running_process_count": 100 + (times.astype(int) % 4),
        "thread_count": 500 + (times.astype(int) % 13),
    })
    # Only required source features are needed by this feature-builder test.
    return frame[["collection_session_id", "timestamp_utc", "elapsed_seconds", *SOURCE_FEATURES]]


class WindowsFeatureTests(unittest.TestCase):
    def test_interpolation_requires_a_bounded_bracket(self) -> None:
        times = np.array([0.0, 1.0, 2.0, 10.0])
        values = np.array([0.0, 2.0, 4.0, 20.0])
        self.assertEqual(_interpolate_at(times, values, 1.5, 2.0), (3.0, 2))
        self.assertIsNone(_interpolate_at(times, values, 5.0, 2.0))

    def test_target_and_lags_stay_inside_each_session(self) -> None:
        first = make_fixture("session-a", "2026-01-01T00:00:00", count=80)
        second = make_fixture("session-b", "2026-01-01T00:10:00", count=80)
        raw = pd.concat([first, second], ignore_index=True)
        result = build_supervised_frame(raw)
        self.assertFalse(result.frame.empty)
        self.assertEqual(set(result.frame["collection_session_id"]), {"session-a", "session-b"})
        self.assertEqual(len(result.feature_columns), 81)
        for _, group in result.frame.groupby("collection_session_id"):
            self.assertGreaterEqual(group["feature_elapsed_seconds"].min(), 30)
            self.assertLessEqual(group["feature_elapsed_seconds"].max(), 50)
            self.assertTrue((group["target_timestamp_utc"] > group["feature_timestamp_utc"]).all())

    def test_chronological_split_embargoes_cross_boundary_targets(self) -> None:
        result = build_supervised_frame(make_fixture("session-a", "2026-01-01T00:00:00", count=300))
        splits, boundary = chronological_split(result.frame)
        train_end = pd.Timestamp(boundary["train_end_utc"])
        validation_end = pd.Timestamp(boundary["validation_end_utc"])
        self.assertTrue((pd.to_datetime(splits["train"]["target_right_timestamp_utc"], utc=True) < train_end).all())
        self.assertTrue((pd.to_datetime(splits["validation"]["target_right_timestamp_utc"], utc=True) < validation_end).all())
        self.assertTrue((pd.to_datetime(splits["validation"]["feature_timestamp_utc"], utc=True) >= train_end).all())
        self.assertTrue((pd.to_datetime(splits["test"]["feature_timestamp_utc"], utc=True) >= validation_end).all())

    def test_latest_inference_features_match_trained_feature_contract(self) -> None:
        raw = make_fixture("session-a", "2026-01-01T00:00:00", count=300)
        supervised = build_supervised_frame(raw).frame
        truncated = raw[raw.elapsed_seconds <= 270].copy()
        current, metadata = build_latest_inference_row(truncated)
        expected = supervised.loc[supervised.feature_elapsed_seconds == 270]
        self.assertEqual(current.shape, (1, 81))
        self.assertEqual(metadata["collection_session_id"], "session-a")
        self.assertEqual(len(expected), 1)
        np.testing.assert_allclose(current.iloc[0].to_numpy(dtype=float), expected.iloc[0][list(current.columns)].to_numpy(dtype=float))

    def test_latest_inference_rejects_insufficient_history(self) -> None:
        raw = make_fixture("session-a", "2026-01-01T00:00:00", count=25)
        with self.assertRaisesRegex(ValueError, "30 seconds"):
            build_latest_inference_row(raw)


if __name__ == "__main__":
    unittest.main()
