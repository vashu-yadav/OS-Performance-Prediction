"""Timestamp-aligned forecasting features for Windows telemetry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TARGETS = {
    "cpu": "cpu_utilization_percent",
    "ram": "ram_utilization_percent",
}
SOURCE_FEATURES = (
    "cpu_utilization_percent",
    "ram_utilization_percent",
    "disk_usage_percent",
    "disk_read_bytes_per_sec",
    "disk_write_bytes_per_sec",
    "network_sent_bytes_per_sec",
    "network_recv_bytes_per_sec",
    "running_process_count",
    "thread_count",
)
LAG_SECONDS = (1, 5, 10, 30)
ROLLING_SECONDS = (10, 30)


@dataclass(frozen=True)
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    source_rows: int
    dropped_for_alignment_or_history: int
    gap_limit_seconds: float


def _interpolate_at(
    times: np.ndarray,
    values: np.ndarray,
    query: float,
    max_gap_seconds: float,
) -> tuple[float, int] | None:
    """Interpolate within a contiguous segment and return right endpoint index."""
    right = int(np.searchsorted(times, query, side="left"))
    if right >= len(times):
        return None
    if np.isclose(times[right], query, atol=1e-9, rtol=0):
        value = values[right]
        return (float(value), right) if np.isfinite(value) else None
    if right == 0:
        return None
    left = right - 1
    gap = times[right] - times[left]
    if gap <= 0 or gap > max_gap_seconds:
        return None
    left_value, right_value = values[left], values[right]
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return None
    fraction = (query - times[left]) / gap
    return float(left_value + fraction * (right_value - left_value)), right


def _value_at_or_before(times: np.ndarray, values: np.ndarray, query: float, max_age_seconds: float) -> float | None:
    """As-of lookup for discrete process/thread counts; never interpolates counts."""
    index = int(np.searchsorted(times, query, side="right")) - 1
    if index < 0 or query - times[index] > max_age_seconds:
        return None
    value = values[index]
    return float(value) if np.isfinite(value) else None


def _segment_groups(frame: pd.DataFrame, gap_limit_seconds: float) -> Iterable[pd.DataFrame]:
    for _, session in frame.groupby("collection_session_id", sort=False):
        ordered = session.sort_values("elapsed_seconds", kind="stable").copy()
        gaps = ordered["elapsed_seconds"].diff().fillna(0)
        ordered["_segment"] = (gaps > gap_limit_seconds).cumsum().astype(int)
        for _, segment in ordered.groupby("_segment", sort=False):
            yield segment.copy()


def build_supervised_frame(
    raw: pd.DataFrame,
    horizon_seconds: float = 30.0,
    gap_limit_seconds: float = 2.0,
    lag_seconds: tuple[int, ...] = LAG_SECONDS,
    rolling_seconds: tuple[int, ...] = ROLLING_SECONDS,
) -> FeatureBuildResult:
    required = {"collection_session_id", "timestamp_utc", "elapsed_seconds", *SOURCE_FEATURES}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Input telemetry is missing required columns: {missing}")
    if horizon_seconds <= 0 or gap_limit_seconds <= 0:
        raise ValueError("horizon_seconds and gap_limit_seconds must be positive")

    source_rows = len(raw)
    result_rows: list[dict[str, object]] = []
    feature_names = [f"current__{name}" for name in SOURCE_FEATURES]
    feature_names.extend(f"lag_{lag}s__{name}" for lag in lag_seconds for name in SOURCE_FEATURES)
    feature_names.extend(
        f"rolling_{window}s_{stat}__{name}"
        for window in rolling_seconds
        for stat in ("mean", "std")
        for name in SOURCE_FEATURES
    )

    for segment in _segment_groups(raw, gap_limit_seconds):
        segment = segment.sort_values("elapsed_seconds", kind="stable").reset_index(drop=True)
        times = segment["elapsed_seconds"].to_numpy(dtype=float)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            continue
        # Rates at the first row after a session/gap describe a longer prior
        # interval and are not comparable to ordinary cadence rates.
        # In a reset segment, the first input row has no trustworthy interval-rate context.
        source = segment.loc[:, SOURCE_FEATURES].astype(float).copy()
        if len(source):
            source.loc[0, [
                "disk_read_bytes_per_sec", "disk_write_bytes_per_sec",
                "network_sent_bytes_per_sec", "network_recv_bytes_per_sec",
            ]] = np.nan
        values_by_name = {name: source[name].to_numpy(dtype=float) for name in SOURCE_FEATURES}
        rolled: dict[tuple[int, str, str], np.ndarray] = {}
        timestamp_ns = pd.to_datetime(segment["timestamp_utc"], utc=True).astype("int64").to_numpy()
        time_index = pd.to_timedelta(times, unit="s")
        for name in SOURCE_FEATURES:
            series = pd.Series(values_by_name[name], index=time_index)
            for window in rolling_seconds:
                rolling = series.rolling(f"{window}s", closed="both", min_periods=2)
                rolled[(window, "mean", name)] = rolling.mean().to_numpy(dtype=float)
                rolled[(window, "std", name)] = rolling.std(ddof=0).to_numpy(dtype=float)

        for i, row in segment.iterrows():
            current_time = times[i]
            cpu_future = _interpolate_at(times, values_by_name[TARGETS["cpu"]], current_time + horizon_seconds, gap_limit_seconds)
            ram_future = _interpolate_at(times, values_by_name[TARGETS["ram"]], current_time + horizon_seconds, gap_limit_seconds)
            if cpu_future is None or ram_future is None:
                continue
            target_right_index = max(cpu_future[1], ram_future[1])
            features: dict[str, float] = {
                f"current__{name}": values_by_name[name][i] for name in SOURCE_FEATURES
            }
            valid_lags = True
            for lag in lag_seconds:
                for name in SOURCE_FEATURES:
                    query = current_time - lag
                    if name in ("running_process_count", "thread_count"):
                        discrete_value = _value_at_or_before(times, values_by_name[name], query, gap_limit_seconds)
                        if discrete_value is None:
                            valid_lags = False
                            break
                        features[f"lag_{lag}s__{name}"] = discrete_value
                    else:
                        value = _interpolate_at(times, values_by_name[name], query, gap_limit_seconds)
                        if value is None:
                            valid_lags = False
                            break
                        features[f"lag_{lag}s__{name}"] = value[0]
                if not valid_lags:
                    break
            if not valid_lags:
                continue
            for window in rolling_seconds:
                for stat in ("mean", "std"):
                    for name in SOURCE_FEATURES:
                        features[f"rolling_{window}s_{stat}__{name}"] = rolled[(window, stat, name)][i]
            if not all(np.isfinite(value) for value in features.values()):
                continue

            target_elapsed = current_time + horizon_seconds
            timestamps = timestamp_ns / 1e9
            target_timestamp = np.interp(target_elapsed, times, timestamps)
            target_right_timestamp = float(timestamps[target_right_index])
            record: dict[str, object] = {
                **features,
                "collection_session_id": row["collection_session_id"],
                "feature_timestamp_utc": pd.to_datetime(row["timestamp_utc"], utc=True),
                "feature_elapsed_seconds": float(current_time),
                "target_timestamp_utc": pd.to_datetime(target_timestamp, unit="s", utc=True),
                "target_right_timestamp_utc": pd.to_datetime(target_right_timestamp, unit="s", utc=True),
                "target_cpu_utilization_percent": cpu_future[0],
                "target_ram_utilization_percent": ram_future[0],
            }
            result_rows.append(record)

    output = pd.DataFrame(result_rows)
    if output.empty:
        return FeatureBuildResult(output, tuple(feature_names), source_rows, source_rows, gap_limit_seconds)
    output = output.sort_values("feature_timestamp_utc", kind="stable").reset_index(drop=True)
    dropped = source_rows - len(output)
    return FeatureBuildResult(output, tuple(feature_names), source_rows, dropped, gap_limit_seconds)


def build_latest_inference_row(
    raw: pd.DataFrame,
    gap_limit_seconds: float = 2.0,
    lag_seconds: tuple[int, ...] = LAG_SECONDS,
    rolling_seconds: tuple[int, ...] = ROLLING_SECONDS,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build one causal feature row at the newest sample without requiring future labels."""
    required = {"collection_session_id", "timestamp_utc", "elapsed_seconds", *SOURCE_FEATURES}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Input telemetry is missing required columns: {missing}")
    if raw.empty:
        raise ValueError("No telemetry rows are available")
    parsed = raw.copy()
    parsed["timestamp_utc"] = pd.to_datetime(parsed["timestamp_utc"], utc=True, errors="coerce")
    parsed["elapsed_seconds"] = pd.to_numeric(parsed["elapsed_seconds"], errors="coerce")
    parsed = parsed.dropna(subset=["timestamp_utc", "elapsed_seconds"])
    if parsed.empty:
        raise ValueError("No valid telemetry timestamps are available")
    latest = parsed.sort_values("timestamp_utc", kind="stable").iloc[-1]
    session = parsed[parsed["collection_session_id"] == latest["collection_session_id"]]
    segments = list(_segment_groups(session, gap_limit_seconds))
    if not segments:
        raise ValueError("The latest sample has no continuous session history")
    segment = max(segments, key=lambda part: float(part["elapsed_seconds"].max()))
    segment = segment.sort_values("elapsed_seconds", kind="stable").reset_index(drop=True)
    times = segment["elapsed_seconds"].to_numpy(dtype=float)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Latest session history contains duplicate or out-of-order elapsed times")
    now = float(latest["elapsed_seconds"])
    if len(times) < 31 or now - float(times[0]) < max(lag_seconds, default=0):
        raise ValueError("At least 30 seconds of uninterrupted history is required for a 30-second forecast")
    if not np.isclose(times[-1], now, atol=1e-6):
        raise ValueError("Latest sample does not belong to the newest continuous segment")

    source = segment.loc[:, SOURCE_FEATURES].astype(float).copy()
    source.loc[0, [
        "disk_read_bytes_per_sec", "disk_write_bytes_per_sec",
        "network_sent_bytes_per_sec", "network_recv_bytes_per_sec",
    ]] = np.nan
    values = {name: source[name].to_numpy(dtype=float) for name in SOURCE_FEATURES}
    index = pd.to_timedelta(times, unit="s")
    features: dict[str, float] = {f"current__{name}": float(values[name][-1]) for name in SOURCE_FEATURES}
    for lag in lag_seconds:
        query = now - lag
        for name in SOURCE_FEATURES:
            if name in ("running_process_count", "thread_count"):
                value = _value_at_or_before(times, values[name], query, gap_limit_seconds)
                if value is None:
                    raise ValueError(f"Insufficient {lag}-second history for {name}")
            else:
                interpolated = _interpolate_at(times, values[name], query, gap_limit_seconds)
                if interpolated is None:
                    raise ValueError(f"Insufficient {lag}-second history for {name}")
                value = interpolated[0]
            features[f"lag_{lag}s__{name}"] = float(value)
    for window in rolling_seconds:
        for stat in ("mean", "std"):
            for name in SOURCE_FEATURES:
                rolling = pd.Series(values[name], index=index).rolling(
                    f"{window}s", closed="both", min_periods=2
                )
                value = float((rolling.mean() if stat == "mean" else rolling.std(ddof=0)).iloc[-1])
                features[f"rolling_{window}s_{stat}__{name}"] = value
    if len(features) != 81 or not np.isfinite(list(features.values())).all():
        raise ValueError("Latest telemetry did not produce all finite trained-model features")
    row = pd.DataFrame([features])
    metadata = {
        "collection_session_id": str(latest["collection_session_id"]),
        "feature_timestamp_utc": latest["timestamp_utc"].isoformat(),
    }
    return row, metadata


def chronological_split(
    supervised: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Split by feature time and embargo labels crossing either boundary."""
    if supervised.empty:
        raise ValueError("No aligned feature/target rows are available")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than 1")
    ordered = supervised.sort_values("feature_timestamp_utc", kind="stable")
    start = pd.Timestamp(ordered["feature_timestamp_utc"].min())
    end = pd.Timestamp(ordered["feature_timestamp_utc"].max())
    span_ns = end.value - start.value
    train_end = pd.Timestamp(start.value + int(span_ns * train_fraction), tz="UTC")
    validation_end = pd.Timestamp(start.value + int(span_ns * (train_fraction + validation_fraction)), tz="UTC")
    t = pd.to_datetime(ordered["feature_timestamp_utc"], utc=True)
    target_right = pd.to_datetime(ordered["target_right_timestamp_utc"], utc=True)

    masks = {
        "train": (t < train_end) & (target_right < train_end),
        "validation": (t >= train_end) & (t < validation_end) & (target_right < validation_end),
        "test": (t >= validation_end),
    }
    splits = {name: ordered.loc[mask].copy().reset_index(drop=True) for name, mask in masks.items()}
    boundaries = {
        "feature_start_utc": start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "validation_end_utc": validation_end.isoformat(),
        "feature_end_utc": end.isoformat(),
        "split_rule": "70/15/15 chronological feature-time cutoffs; train and validation rows are retained only when the rightmost source row used to interpolate their future target is before the next split boundary.",
    }
    return splits, boundaries
