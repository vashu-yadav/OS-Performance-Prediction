"""Collect timestamped Windows host telemetry into an append-only CSV."""
from __future__ import annotations

import argparse
import csv
import ctypes
import os
import sys
import time
import uuid
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ctypes import wintypes

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "windows_telemetry_workload.csv"
WORKLOAD_LABELS = ("Idle", "Gaming", "Browsing", "Development", "Multitasking")

COLUMNS = (
    "collection_session_id",
    "timestamp_utc",
    "elapsed_seconds",
    "cpu_utilization_percent",
    "ram_utilization_percent",
    "available_ram_bytes",
    "disk_usage_percent",
    "disk_read_bytes_total",
    "disk_write_bytes_total",
    "disk_read_bytes_per_sec",
    "disk_write_bytes_per_sec",
    "network_sent_bytes_total",
    "network_recv_bytes_total",
    "network_sent_bytes_per_sec",
    "network_recv_bytes_per_sec",
    "running_process_count",
    "thread_count",
    "workload_label",
)

PROCESS_COLUMNS = (
    "collection_session_id", "timestamp_utc", "workload_label", "pid", "process_name",
    "cpu_percent", "memory_percent", "cpu_rank", "memory_rank",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval-seconds", type=float, default=1.0,
        help="Requested time between samples (default: 1 second).",
    )
    parser.add_argument(
        "--duration-seconds", type=float, default=300.0,
        help="Collection duration (default: 300 seconds).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"CSV destination (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--workload-label", choices=WORKLOAD_LABELS,
        help="Optional context label applied to this collection session.",
    )
    parser.add_argument("--top-processes", type=int, default=5,
                        help="Number of top CPU/memory processes to record per sample (default: 5).")
    parser.add_argument("--process-sampling-interval-seconds", type=float, default=60.0,
                        help="Background process enumeration interval (default: 60 seconds to limit overhead).")
    parser.add_argument("--process-output", type=Path,
                        help="Optional process-detail CSV path; defaults beside --output.")
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be greater than zero")
    if args.top_processes <= 0:
        parser.error("--top-processes must be greater than zero")
    if args.process_sampling_interval_seconds <= 0:
        parser.error("--process-sampling-interval-seconds must be greater than zero")
    return args


def _require_windows() -> None:
    if os.name != "nt":
        raise OSError("Windows telemetry collection is supported only on Windows.")


def _read_counter(counters: Any, key: str) -> int | None:
    return None if counters is None else getattr(counters, key, None)


def _rate(previous: int | None, current: int | None, elapsed: float) -> float | None:
    """Return a counter rate; keep unavailable/reset counters missing, not zero."""
    if previous is None or current is None or elapsed <= 0 or current < previous:
        return None
    return (current - previous) / elapsed


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]


def _process_counts() -> tuple[int, int]:
    """Return the active PID count from psutil and Windows' thread total."""
    _require_windows()
    information = _PerformanceInformation()
    information.cb = ctypes.sizeof(information)
    get_performance_info = ctypes.WinDLL("psapi.dll", use_last_error=True).GetPerformanceInfo
    get_performance_info.argtypes = [ctypes.POINTER(_PerformanceInformation), wintypes.DWORD]
    get_performance_info.restype = wintypes.BOOL
    if not get_performance_info(ctypes.byref(information), information.cb):
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "GetPerformanceInfo failed")
    return len(psutil.pids()), int(information.ThreadCount)


def _system_drive() -> str:
    drive = os.environ.get("SystemDrive", "C:")
    return drive.rstrip("\\/") + "\\"


def _snapshot(
    *,
    started_at: float,
    session_id: str,
    previous_disk: Any,
    previous_network: Any,
    interval_elapsed: float,
    workload_label: str | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    # Timestamp near the end so each row describes the completed snapshot.
    process_count, thread_count = _process_counts()
    disk = psutil.disk_io_counters(perdisk=False)
    network = psutil.net_io_counters(pernic=False)
    memory = psutil.virtual_memory()
    try:
        disk_usage = psutil.disk_usage(_system_drive()).percent
    except (OSError, PermissionError):
        disk_usage = None

    disk_read = _read_counter(disk, "read_bytes")
    disk_write = _read_counter(disk, "write_bytes")
    network_sent = _read_counter(network, "bytes_sent")
    network_received = _read_counter(network, "bytes_recv")

    row: dict[str, Any] = {
        "collection_session_id": session_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "cpu_utilization_percent": psutil.cpu_percent(interval=None),
        "ram_utilization_percent": memory.percent,
        "available_ram_bytes": memory.available,
        "disk_usage_percent": disk_usage,
        "disk_read_bytes_total": disk_read,
        "disk_write_bytes_total": disk_write,
        "disk_read_bytes_per_sec": _rate(_read_counter(previous_disk, "read_bytes"), disk_read, interval_elapsed),
        "disk_write_bytes_per_sec": _rate(_read_counter(previous_disk, "write_bytes"), disk_write, interval_elapsed),
        "network_sent_bytes_total": network_sent,
        "network_recv_bytes_total": network_received,
        "network_sent_bytes_per_sec": _rate(_read_counter(previous_network, "bytes_sent"), network_sent, interval_elapsed),
        "network_recv_bytes_per_sec": _rate(_read_counter(previous_network, "bytes_recv"), network_received, interval_elapsed),
        "running_process_count": process_count,
        "thread_count": thread_count,
        "workload_label": workload_label or "",
    }
    return row, disk, network


def _sample_top_processes(
    session_id: str,
    timestamp_utc: str,
    workload_label: str | None,
    top_n: int,
) -> list[dict[str, Any]]:
    """Collect only process name and resource use; never inspect cmdline or user data."""
    samples: list[dict[str, Any]] = []
    for process in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"], ad_value=None):
        try:
            info = process.info
            if not info.get("name") or info.get("cpu_percent") is None or info.get("memory_percent") is None:
                continue
            samples.append({
                "pid": int(info["pid"]),
                "process_name": str(info["name"]),
                "cpu_percent": float(info["cpu_percent"]),
                "memory_percent": float(info["memory_percent"]),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            continue

    cpu_top = sorted(samples, key=lambda item: item["cpu_percent"], reverse=True)[:top_n]
    memory_top = sorted(samples, key=lambda item: item["memory_percent"], reverse=True)[:top_n]
    cpu_ranks = {item["pid"]: rank for rank, item in enumerate(cpu_top, 1)}
    memory_ranks = {item["pid"]: rank for rank, item in enumerate(memory_top, 1)}
    selected = {item["pid"]: item for item in (*cpu_top, *memory_top)}
    return [
        {
            "collection_session_id": session_id,
            "timestamp_utc": timestamp_utc,
            "workload_label": workload_label or "",
            **item,
            "cpu_rank": cpu_ranks.get(pid, ""),
            "memory_rank": memory_ranks.get(pid, ""),
        }
        for pid, item in selected.items()
    ]


def _prime_process_cpu() -> None:
    """Initialize psutil process CPU counters before the first timed sample."""
    for process in psutil.process_iter(attrs=["pid"], ad_value=None):
        try:
            process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            continue


def _open_process_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_content = path.is_file() and path.stat().st_size > 0
    handle = path.open("a+", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=PROCESS_COLUMNS, extrasaction="raise")
    if has_content:
        handle.seek(0)
        header = next(csv.reader([handle.readline().rstrip("\r\n")]), [])
        if tuple(header) != PROCESS_COLUMNS:
            handle.close()
            raise ValueError(f"Existing process CSV header does not match schema: {path}")
        handle.seek(0, os.SEEK_END)
    else:
        writer.writeheader()
        handle.flush()
    return handle, writer


def _open_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists_with_content = path.is_file() and path.stat().st_size > 0
    handle = path.open("a+", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
    if exists_with_content:
        handle.seek(0)
        existing_header = next(csv.reader([handle.readline().rstrip("\r\n")]), [])
        if tuple(existing_header) != COLUMNS:
            handle.close()
            raise ValueError(f"Existing CSV header does not match telemetry schema: {path}")
        handle.seek(0, os.SEEK_END)
    else:
        writer.writeheader()
        handle.flush()
    return handle, writer


def collect_telemetry(
    output_path: Path = DEFAULT_OUTPUT,
    interval_seconds: float = 1.0,
    duration_seconds: float = 300.0,
    workload_label: str | None = None,
    top_processes: int = 5,
    process_sampling_interval_seconds: float = 60.0,
    process_output_path: Path | None = None,
) -> int:
    """Collect live host samples and append them to a schema-checked CSV."""
    _require_windows()
    if interval_seconds <= 0 or duration_seconds <= 0 or top_processes <= 0 or process_sampling_interval_seconds <= 0:
        raise ValueError("collection intervals, duration, and top_processes must be greater than zero")
    if workload_label is not None and workload_label not in WORKLOAD_LABELS:
        raise ValueError(f"workload_label must be one of {WORKLOAD_LABELS}")

    # psutil's first non-blocking CPU measurement is a warm-up and is discarded.
    psutil.cpu_percent(interval=None)
    _prime_process_cpu()
    previous_disk = psutil.disk_io_counters(perdisk=False)
    previous_network = psutil.net_io_counters(pernic=False)
    started = time.monotonic()
    deadline = started + duration_seconds
    session_id = uuid.uuid4().hex
    next_sample_at = started + interval_seconds
    last_sample_at = started
    rows_written = 0
    last_process_sample_at = started - process_sampling_interval_seconds

    handle, writer = _open_writer(Path(output_path))
    process_path = process_output_path or Path(output_path).with_name(f"{Path(output_path).stem}_processes.csv")
    try:
        process_handle, process_writer = _open_process_writer(Path(process_path))
    except Exception:
        handle.close()
        raise
    process_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="os-telemetry-processes")
    process_future: Future[list[dict[str, Any]]] | None = None
    try:
        while next_sample_at <= deadline:
            delay = next_sample_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            sampled_at = time.monotonic()
            if sampled_at > deadline:
                break
            interval_elapsed = sampled_at - last_sample_at
            if interval_elapsed <= 0:
                next_sample_at += interval_seconds
                continue
            row, previous_disk, previous_network = _snapshot(
                started_at=started,
                session_id=session_id,
                previous_disk=previous_disk,
                previous_network=previous_network,
                interval_elapsed=interval_elapsed,
                workload_label=workload_label,
            )
            writer.writerow(row)
            handle.flush()
            if process_future is not None and process_future.done():
                try:
                    process_writer.writerows(process_future.result())
                    process_handle.flush()
                except Exception as exc:
                    warnings.warn(f"Skipping a failed process telemetry sample: {exc}", RuntimeWarning)
                process_future = None
            if process_future is None and sampled_at - last_process_sample_at >= process_sampling_interval_seconds:
                process_future = process_executor.submit(
                    _sample_top_processes, session_id, str(row["timestamp_utc"]), workload_label, top_processes
                )
                last_process_sample_at = sampled_at
            rows_written += 1
            last_sample_at = sampled_at
            next_sample_at += interval_seconds
            # Do not add sampling time on top of snapshot cost. If enumeration
            # was slower than the interval, skip missed ticks and preserve cadence.
            finished_at = time.monotonic()
            while next_sample_at <= finished_at:
                next_sample_at += interval_seconds
    finally:
        if process_future is not None:
            try:
                process_writer.writerows(process_future.result())
                process_handle.flush()
            except Exception as exc:
                warnings.warn(f"Skipping a failed final process telemetry sample: {exc}", RuntimeWarning)
        process_executor.shutdown(wait=True, cancel_futures=True)
        handle.close()
        process_handle.close()
    return rows_written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = collect_telemetry(
            output_path=args.output,
            interval_seconds=args.interval_seconds,
            duration_seconds=args.duration_seconds,
            workload_label=args.workload_label,
            top_processes=args.top_processes,
            process_sampling_interval_seconds=args.process_sampling_interval_seconds,
            process_output_path=args.process_output,
        )
    except KeyboardInterrupt:
        print("Collection interrupted; completed samples were saved.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"Telemetry collection failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {count} live Windows telemetry samples to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

