"""Non-blocking snapshots of Windows host health for the live dashboard."""
from __future__ import annotations

import os
import time
from typing import Any

import psutil


def sample_live_metrics(previous: dict[str, Any] | None = None) -> dict[str, Any]:
    if os.name != "nt":
        raise OSError("Live system monitoring is supported only on Windows.")
    now = time.monotonic()
    cpu = float(psutil.cpu_percent(interval=None))
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.environ.get("SystemDrive", "C:") + "\\")
    net = psutil.net_io_counters(pernic=False)
    disk_io = psutil.disk_io_counters(perdisk=False)
    from os_performance_prediction.telemetry import _process_counts

    process_count, thread_count = _process_counts()
    elapsed = now - previous["monotonic"] if previous else 0.0

    def rate(current: int | None, key: str) -> float | None:
        if current is None or not previous or elapsed <= 0:
            return None
        old = previous.get(key)
        if old is None or current < old:
            return None
        return (current - old) / elapsed

    cpu_status = "High" if cpu >= 85 else "Elevated" if cpu >= 70 else "Normal"
    ram_status = "High" if memory.percent >= 90 else "Elevated" if memory.percent >= 80 else "Normal"
    disk_status = "High" if disk.percent >= 95 else "Elevated" if disk.percent >= 85 else "Normal"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "monotonic": now,
        "cpu_percent": cpu,
        "ram_percent": float(memory.percent),
        "available_ram_bytes": int(memory.available),
        "disk_usage_percent": float(disk.percent),
        "network_sent_bytes_per_sec": rate(net.bytes_sent, "network_sent_bytes"),
        "network_recv_bytes_per_sec": rate(net.bytes_recv, "network_recv_bytes"),
        "disk_read_bytes_per_sec": rate(getattr(disk_io, "read_bytes", None), "disk_read_bytes"),
        "disk_write_bytes_per_sec": rate(getattr(disk_io, "write_bytes", None), "disk_write_bytes"),
        "network_sent_bytes": net.bytes_sent,
        "network_recv_bytes": net.bytes_recv,
        "disk_read_bytes": getattr(disk_io, "read_bytes", None),
        "disk_write_bytes": getattr(disk_io, "write_bytes", None),
        "process_count": process_count,
        "thread_count": thread_count,
        "cpu_health": cpu_status,
        "ram_health": ram_status,
        "disk_health": disk_status,
    }
