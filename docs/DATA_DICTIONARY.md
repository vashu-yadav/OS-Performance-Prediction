# Windows Telemetry Data Dictionary

## Active dataset

| Property | Description |
|---|---|
| Dataset | Self-collected Windows host telemetry |
| Collection library | `psutil` 7.0.0 in the project's Python 3.12.10 `.venv` |
| Output | `data/raw/windows_telemetry.csv` |
| Observation grain | One sample per requested interval; actual elapsed time is recorded because scheduling/collection overhead can cause drift |
| Scope | One Windows computer, observed only during the sessions collected; not representative of all Windows systems |
| Provenance | Directly measured from this Windows host at collection time; not synthetic and not imported from Google Cluster Data |

## Fields

| Column | Type in CSV | Description | Units / interpretation |
|---|---|---|---|
| `collection_session_id` | String | Unique identifier for one invocation of the collector | UUID hex; group this field to prevent lag/target windows crossing sessions |
| `timestamp_utc` | ISO-8601 string | Sample timestamp in UTC | Millisecond precision, `Z` suffix |
| `elapsed_seconds` | Float | Seconds from the start of this collector invocation | Seconds; resets for each collection invocation, so use timestamp for chronology across appended sessions |
| `cpu_utilization_percent` | Float | Overall CPU utilization reported by `psutil.cpu_percent` | Percent, 0â€“100; first warm-up measurement is discarded |
| `ram_utilization_percent` | Float | Virtual memory utilization reported by `psutil.virtual_memory().percent` | Percent, 0â€“100 |
| `available_ram_bytes` | Integer | Memory estimated available without swapping | Bytes |
| `disk_usage_percent` | Float or blank | Used capacity percentage for the Windows system drive | Percent full, not disk I/O busy time |
| `disk_read_bytes_total` | Integer or blank | Cumulative disk reads reported by OS since boot | Bytes; may be unavailable |
| `disk_write_bytes_total` | Integer or blank | Cumulative disk writes reported by OS since boot | Bytes; may be unavailable |
| `disk_read_bytes_per_sec` | Float or blank | Change in cumulative disk reads since previous sample divided by actual elapsed sample time | Bytes/second; blank if counters are unavailable/reset |
| `disk_write_bytes_per_sec` | Float or blank | Change in cumulative disk writes since previous sample divided by actual elapsed sample time | Bytes/second; blank if counters are unavailable/reset |
| `network_sent_bytes_total` | Integer or blank | Cumulative network bytes sent across interfaces since boot | Bytes; no packet contents are collected |
| `network_recv_bytes_total` | Integer or blank | Cumulative network bytes received across interfaces since boot | Bytes; no packet contents are collected |
| `network_sent_bytes_per_sec` | Float or blank | Change in cumulative sent bytes since previous sample divided by actual elapsed sample time | Bytes/second; blank if counters are unavailable/reset |
| `network_recv_bytes_per_sec` | Float or blank | Change in cumulative received bytes since previous sample divided by actual elapsed sample time | Bytes/second; blank if counters are unavailable/reset |
| `running_process_count` | Integer | Number of active process IDs returned by `psutil.pids()` during the snapshot | Count; processes can start/exit while the PID list is read |
| `thread_count` | Integer | System-wide active thread count returned by the Windows `GetPerformanceInfo` API | Count |

The CSV header is the authoritative schema. The collector refuses to append to an existing file with a different header. Blank values represent unavailable measurements and must not be silently replaced with zero.

## Collection parameters

`--interval-seconds` controls the requested sampling interval (default 1 second). `--duration-seconds` controls wall-clock collection duration (default 300 seconds). Data is appended to the configured destination; each invocation has a unique session ID and begins a new elapsed-time origin. Sampling uses monotonic time for interval/rate calculations and UTC wall-clock timestamps for persisted chronology.

## Implemented targets and feature contract

Targets are `cpu_utilization_percent` and `ram_utilization_percent` 30 seconds after each feature timestamp, aligned only within the same session and continuous segment. The feature builder interpolates by elapsed time and starts a new segment after gaps over two seconds. Current, lagged, and trailing 10/30-second mean/std features use only readings at or before prediction time. Feature diagnostics are calculated from training rows only. Chronological 70/15/15 feature-time cutoffs are used; rows are embargoed if the rightmost observation used to interpolate the target reaches the next partition boundary.

The current engineered dataset has 1,559 rows: 1,060 train, 204 validation, and 234 test. 161 raw rows lack valid aligned history/target coverage. The operational exploratory floor (500/100/100) is not a statistical guarantee.

## Current dataset validation summary

Full profile: `results/windows_telemetry_profile.json`. The CSV has 1,720 rows and exactly 17 expected fields, with no missing values, invalid timestamps, duplicate full rows, or duplicate session/elapsed keys. It contains a five-row smoke session and a 1,715-row session spanning 1,798.968 elapsed seconds. Median cadence is 1.0 second; the main session has two gaps above two seconds, maximum 48.75 seconds. There are no constant numeric columns. `disk_usage_percent` has only 20 unique values over a narrow 65.0?67.7% range and is weakly varying. RAM utilization and available RAM correlate at -0.999985; available RAM is excluded as redundant. Cumulative counters are excluded as time/history proxies; rates are used.

Raw distributions (mean; standard deviation; min?median?max): CPU utilization 31.71%; 13.46; 1.4?31.2?93.2%. RAM utilization 89.44%; 5.29; 80.6?89.2?99.9%. Disk capacity usage 66.56%; 0.79; 65.0?67.1?67.7%. Available RAM averages 1.77 billion bytes and ranges from 21.9 million to 3.25 billion bytes. The profile JSON has percentiles and distributions for every numeric field.

## Current model-data limitations

The file clears the operational floor for an exploratory fit, but almost all supervised rows come from one continuous 1,620-row segment in one 30-minute session. No workload regimes were explicitly labeled, and held-out CPU/RAM performance is weak. As a practical next collection (not a statistical guarantee), gather at least four separate 60-minute sessions over at least two days spanning idle, typical use, CPU-heavy, and memory-heavy conditions, with context recorded. This one-machine dataset does not establish cross-device generalization.

## Missingness, measurement limits, and risks

- `psutil`/Windows counters may be unavailable on some devices; absent values stay blank.
- Cumulative I/O counters can reset at reboot; rate for that transition stays blank.
- The PID list is a moving snapshot; processes can start or exit while the count is read.
- Collection overhead and system load can make sampling intervals irregular; use timestamps and elapsed intervals.
- Disk capacity percentage is not disk saturation/latency. The initial schema does not claim a disk busy percentage.
- Data contains no deliberate test workload labels. Collection session context must be noted separately if experiments compare idle, browsing, development, or other conditions.
- A one-machine dataset cannot establish cross-device or population-level generalization.
- The current training results are exploratory; more varied, documented collection sessions are needed for stronger conclusions.

## Google Cluster Data 2011-2 alternative (not used)

The original proposal to use Google's `clusterdata-2011-2` trace is **pending/abandoned as the active dataset** because access approval was delayed. No Google Cluster Data has been downloaded or used in this project. It is historical Borg cluster workload data, not Windows host telemetry. Its official documentation remains at [Google's 2011-2 trace page](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md); preserve it only as a possible future alternative if access is granted and the research scope is explicitly revised. Do not combine it with the Windows host CSV or claim that it validates this machine's predictions.


## Workload-aware collection extension

The existing 17-column Phase 3 file is preserved. New collector runs default to `data/raw/windows_telemetry_workload.csv`, whose aggregate schema adds `workload_label` (string; optional; one of Idle, Gaming, Browsing, Development, Multitasking). This file is intentionally separate so the original dataset, evaluation, and trained artifacts are not rewritten. The feature builder accepts the extra context column but does not use workload label as a model input; there is no evidence yet that labels improve prediction.

A second sidecar, `data/raw/windows_telemetry_workload_processes.csv`, records a union of the top-N CPU and memory processes at the configured process-sampling interval (default 60 seconds, sampled in a background worker). Fields are session ID, timestamp, workload label, PID, process name, process CPU percentage, memory percentage, CPU rank, and memory rank. Process CPU use is measured between samples and may exceed 100% on multicore systems. Process enumeration is throttled and runs in a background thread to avoid blocking aggregate sampling; one timing check on this host (336 visible PIDs) took about 0.95 seconds, but the cost depends on process count and machine load. Short-lived processes and inaccessible process fields may be missed. No command-line arguments, usernames, executable paths, or window titles are read. Process names and PIDs can still be sensitive; keep the sidecar local and remove it before sharing if needed.

As of this implementation, no persistent genuine workload-labeled collection has yet been made; the new collector fields were tested with brief temporary live captures. Workload category statistics in the dashboard are descriptive, user-labeled, and potentially unbalanced. They are not controlled experiments and cannot show causal workload effects. The original unlabeled CSV remains valid for the Phase 3 model/evaluation; new workload samples can be analyzed separately but do not change the saved model metrics.

