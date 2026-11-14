# Local telemetry files

Telemetry CSVs in this directory are generated from the local Windows computer and are intentionally excluded from Git. The dashboard uses these files for history, workload analysis, live forecast features, and local process observations.

After cloning the repository, collect a session from PowerShell, for example:

```powershell
.\.venv\Scripts\python.exe src\os_performance_prediction\telemetry.py --interval-seconds 1 --duration-seconds 1800 --workload-label Development
```

This creates `windows_telemetry_workload.csv` and a `windows_telemetry_workload_processes.csv` sidecar. Keep those files local; process names and resource timelines can be sensitive. See the project README and data dictionary for details.
