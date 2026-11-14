"""Profile the complete self-collected Windows telemetry CSV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from os_performance_prediction.runtime_guard import reject_streamlit_pipeline_entrypoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "windows_telemetry.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "windows_telemetry_profile.json"
EXPECTED_COLUMNS = (
    "collection_session_id", "timestamp_utc", "elapsed_seconds",
    "cpu_utilization_percent", "ram_utilization_percent", "available_ram_bytes",
    "disk_usage_percent", "disk_read_bytes_total", "disk_write_bytes_total",
    "disk_read_bytes_per_sec", "disk_write_bytes_per_sec",
    "network_sent_bytes_total", "network_recv_bytes_total",
    "network_sent_bytes_per_sec", "network_recv_bytes_per_sec",
    "running_process_count", "thread_count",
)


def analyze_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry CSV not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(EXPECTED_COLUMNS) - {"workload_label"})
    if missing_columns:
        raise ValueError(f"Telemetry schema is missing columns: {missing_columns}")
    if extra_columns:
        raise ValueError(f"Telemetry schema has unexpected columns: {extra_columns}")

    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    numeric = frame.select_dtypes(include="number")
    distributions: dict[str, Any] = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99]) if len(values) else pd.Series(dtype=float)
        distributions[column] = {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size > 1 else None,
            "min": float(values.min()) if values.size else None,
            "p01": float(quantiles.loc[0.01]) if len(quantiles) else None,
            "p25": float(quantiles.loc[0.25]) if len(quantiles) else None,
            "median": float(quantiles.loc[0.5]) if len(quantiles) else None,
            "p75": float(quantiles.loc[0.75]) if len(quantiles) else None,
            "p99": float(quantiles.loc[0.99]) if len(quantiles) else None,
            "max": float(values.max()) if values.size else None,
            "unique": int(values.nunique()),
        }

    columns = []
    for column in frame.columns:
        series = frame[column]
        columns.append({
            "name": column,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        })

    gaps_by_session: dict[str, Any] = {}
    segments_by_session: dict[str, list[int]] = {}
    elapsed_timestamp_errors: list[float] = []
    for session_id, group in frame.assign(_parsed_timestamp=timestamps).groupby("collection_session_id", sort=False):
        ordered = group.sort_values("elapsed_seconds", kind="stable")
        elapsed_gaps = ordered["elapsed_seconds"].diff().dropna()
        time_gaps = ordered["_parsed_timestamp"].diff().dt.total_seconds().dropna()
        cadence = float(elapsed_gaps[elapsed_gaps > 0].median()) if (elapsed_gaps > 0).any() else None
        gap_limit = max(2.0, 2.0 * cadence) if cadence else 2.0
        breaks = ordered["elapsed_seconds"].diff().fillna(0) > gap_limit
        segment_ids = breaks.cumsum()
        segment_lengths = [int(n) for n in ordered.groupby(segment_ids).size().tolist()]
        elapsed_timestamp_errors.extend((time_gaps.to_numpy() - elapsed_gaps.to_numpy()).tolist())
        key = str(session_id)
        gaps_by_session[key] = {
            "rows": int(len(ordered)),
            "elapsed_start_seconds": float(ordered["elapsed_seconds"].min()),
            "elapsed_end_seconds": float(ordered["elapsed_seconds"].max()),
            "elapsed_span_seconds": float(ordered["elapsed_seconds"].max() - ordered["elapsed_seconds"].min()),
            "timestamp_start_utc": ordered["_parsed_timestamp"].min().isoformat(),
            "timestamp_end_utc": ordered["_parsed_timestamp"].max().isoformat(),
            "cadence_median_seconds": cadence,
            "gap_limit_seconds_for_segmentation": gap_limit,
            "gap_count": int(len(elapsed_gaps)),
            "gap_over_2_seconds": int((elapsed_gaps > 2).sum()),
            "gap_over_5_seconds": int((elapsed_gaps > 5).sum()),
            "gap_max_seconds": float(elapsed_gaps.max()) if len(elapsed_gaps) else None,
            "timestamp_gap_median_seconds": float(time_gaps.median()) if len(time_gaps) else None,
            "timestamp_elapsed_difference_abs_max_seconds": float(np.max(np.abs(time_gaps.to_numpy() - elapsed_gaps.to_numpy()))) if len(time_gaps) else None,
            "contiguous_segment_row_counts": segment_lengths,
        }
        segments_by_session[key] = segment_lengths

    duplicate_key = int(frame.duplicated(["collection_session_id", "elapsed_seconds"]).sum())
    duplicate_rows = int(frame.duplicated().sum())
    const = [c for c in numeric if numeric[c].nunique(dropna=True) <= 1]
    near_constant = []
    for c in numeric:
        s = numeric[c].dropna()
        if len(s) and c not in const and s.nunique() <= 20:
            near_constant.append({"column": c, "unique_values": int(s.nunique()), "min": float(s.min()), "max": float(s.max())})

    raw_candidate_features = [
        "cpu_utilization_percent", "ram_utilization_percent", "disk_usage_percent",
        "disk_read_bytes_per_sec", "disk_write_bytes_per_sec",
        "network_sent_bytes_per_sec", "network_recv_bytes_per_sec",
        "running_process_count", "thread_count",
    ]
    corr = frame[raw_candidate_features].corr()
    high_corr = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            r = corr.loc[left, right]
            if pd.notna(r) and abs(r) >= 0.98:
                high_corr.append({"feature_a": left, "feature_b": right, "pearson_r": float(r)})
    ram_available_corr = frame[["ram_utilization_percent", "available_ram_bytes"]].corr().iloc[0, 1]

    # Estimate exact 30-second labels by local interpolation within uninterrupted
    # sections. This is a coverage diagnostic only, not training or evaluation.
    aligned_targets = 0
    target_candidates = 0
    for session_id, group in frame.groupby("collection_session_id", sort=False):
        ordered = group.sort_values("elapsed_seconds", kind="stable")
        times = ordered["elapsed_seconds"].to_numpy(dtype=float)
        cadence = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        gap_limit = max(2.0, 2.0 * cadence)
        segment = np.cumsum(np.r_[False, np.diff(times) > gap_limit])
        for seg_id in np.unique(segment):
            positions = np.flatnonzero(segment == seg_id)
            seg_times = times[positions]
            queries = seg_times + 30.0
            right = np.searchsorted(seg_times, queries, side="left")
            valid = (right < len(seg_times)) & (queries <= seg_times[-1])
            target_candidates += int(valid.sum())
            for j in np.flatnonzero(valid):
                k = int(right[j])
                if k == 0:
                    continue
                if seg_times[k] == queries[j] or seg_times[k] - seg_times[k - 1] <= gap_limit:
                    aligned_targets += 1

    profile = {
        "dataset_path": str(path.resolve()),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "expected_columns": list(EXPECTED_COLUMNS),
        "missing_expected_columns": missing_columns,
        "unexpected_columns": extra_columns,
        "columns": columns,
        "missing_values": {c: int(n) for c, n in frame.isna().sum().items() if int(n) > 0},
        "invalid_timestamps": invalid_timestamps,
        "duplicate_full_rows": duplicate_rows,
        "duplicate_session_elapsed_rows": duplicate_key,
        "session_count": int(frame["collection_session_id"].nunique()),
        "workload_label_counts": frame["workload_label"].fillna("").replace("", "Unlabeled").value_counts().to_dict() if "workload_label" in frame else {"Unlabeled": int(len(frame))},
        "session_rows": {str(k): int(v) for k, v in frame.groupby("collection_session_id", sort=False).size().items()},
        "timestamp_start_utc": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "timestamp_end_utc": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "timestamp_gap_seconds_by_session": gaps_by_session,
        "numeric_distributions": distributions,
        "constant_numeric_columns": const,
        "near_constant_numeric_columns": near_constant,
        "high_correlation_candidate_feature_pairs_abs_ge_0_98": high_corr,
        "ram_percent_vs_available_bytes_pearson_r": float(ram_available_corr),
        "candidate_30s_target_alignment": {
            "aligned_rows_using_local_interpolation": aligned_targets,
            "candidate_rows_with_30s_future_inside_session": target_candidates,
            "gap_limit_seconds": 2.0,
            "note": "Diagnostic alignment count only. Labels are generated later by the feature pipeline; this is not a metric or model result.",
        },
        "feature_reliability_notes": [
            "Session IDs, timestamps, and elapsed time are split/ordering metadata, not model features.",
            "Cumulative I/O/network totals are excluded from forecasting features because they encode time and machine history; per-second rates are used instead.",
            "available_ram_bytes is nearly a deterministic inverse of RAM utilization on this host and is excluded as a redundant feature.",
            "disk_usage_percent changes little in this capture; retain only if train-only feature diagnostics show useful variation.",
            "Counter rates measured across unusually long sampling gaps have different averaging windows; feature segments reset after gaps exceeding the cadence threshold.",
        ],
        "elapsed_timestamp_step_difference_abs_max_seconds": max((abs(v) for v in elapsed_timestamp_errors), default=None),
    }
    return profile


def main() -> int:
    if reject_streamlit_pipeline_entrypoint("Telemetry profiling"):
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = analyze_dataset(args.data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("columns", "numeric_distributions")}, indent=2))
    print("Column profiles:")
    print(json.dumps(report["columns"], indent=2))
    print("Numeric distributions:")
    print(json.dumps(report["numeric_distributions"], indent=2))
    print(f"Saved profile: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
