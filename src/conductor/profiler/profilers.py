from conductor.profiler.time_profiler import TimeProfiler
from conductor.profiler.system_monitor import SystemMonitor
from conductor.profiler.powercap_profiler import PowercapProfiler


profilers = {
    "time_profiler": TimeProfiler,
    "system_monitor": SystemMonitor,
    "powercap_profiler": PowercapProfiler
}