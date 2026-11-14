"""Validate an available Google Cluster Data task_usage CSV without inventing schema.

Expected headers are supplied from an authentic file/schema via --expected-columns.
The default path is data/raw/task_usage.csv; CSV files may be gzip compressed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=project_root / "data" / "raw" / "task_usage.csv",
        help="Path to an authentic task_usage CSV or .csv.gz file.",
    )
    parser.add_argument(
        "--expected-columns",
        nargs="+",
        help="Exact header names copied from the official schema and/or source file.",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="CSV delimiter (default: comma); set from the real file format.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Maximum number of data rows to inspect (default: 1000).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "results" / "dataset_validation.json",
        help="JSON report destination.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset.resolve()),
        "sample_rows_requested": args.rows,
        "delimiter": args.sep,
        "expected_columns_supplied": args.expected_columns is not None,
        "expected_columns": args.expected_columns,
        "status": "not_checked",
    }
    if args.rows < 1:
        report["status"] = "invalid_arguments"
        report["error"] = "--rows must be at least 1"
        return write_report(args.report, report, 2)
    if not args.dataset.is_file():
        report["status"] = "dataset_missing"
        report["error"] = (
            "No dataset file is available at this path. Obtain the official "
            "Google Cluster Data 2011-2 files through Google's documented access process."
        )
        return write_report(args.report, report, 2)

    try:
        import pandas as pd

        frame = pd.read_csv(
            args.dataset,
            sep=args.sep,
            nrows=args.rows,
            low_memory=False,
        )
        actual = [str(column) for column in frame.columns]
        expected = args.expected_columns
        missing_columns = sorted(set(expected or []) - set(actual))
        report.update(
            {
                "status": "schema_validated" if expected is not None and not missing_columns else (
                    "schema_mismatch" if expected is not None else "schema_unverified"
                ),
                "actual_columns": actual,
                "missing_expected_columns": missing_columns,
                "inspected_rows": int(len(frame)),
                "dtypes": {str(name): str(dtype) for name, dtype in frame.dtypes.items()},
                "missing_values_in_sample": {
                    str(name): int(count)
                    for name, count in frame.isna().sum().items()
                    if int(count) > 0
                },
            }
        )
        exit_code = 0 if expected is not None and not missing_columns else 1
    except Exception as exc:  # report parser/encoding/format failures to the caller
        report["status"] = "read_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 2
    return write_report(args.report, report, exit_code)


def write_report(path: Path, report: dict[str, object], exit_code: int) -> int:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"Could not write validation report: {exc}", file=sys.stderr)
        exit_code = 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
