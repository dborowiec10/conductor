
from collections import OrderedDict
from unittest import result
from conductor.profiler.profiler import Profiler
from conductor._base import Configurable
from conductor.experiment import Experiment
from multiprocessing import Process, Event, Pipe
import time
import psutil
from pynvml import *

import logging
logger = logging.getLogger("conductor.profiler.system_monitor")

class SystemMonitorCollector(Process):
    _name = "SystemMonitorCollector"

    def __repr__(self):
        return SystemMonitorCollector._name

    metrics_collectors = {
        "cpu": {
            "times_perc": {
                "host_cpu_util_user": lambda x: x.user,
                "host_cpu_util_nice": lambda x: x.nice,
                "host_cpu_util_system": lambda x: x.system,
                "host_cpu_util_idle": lambda x: x.idle,
                "host_cpu_util_iowait": lambda x: x.iowait,
                "host_cpu_util_irq": lambda x: x.irq,
                "host_cpu_util_softirq": lambda x: x.softirq
            },
            "others": {
                "host_cpu_freq": lambda x: psutil.cpu_freq().current,
                "host_cpu_util_perc": lambda x: psutil.cpu_percent(interval=None),
                "host_mem_prc": lambda x: psutil.virtual_memory().percent
            }
        },
        "gpu": {
            "memory": {
                "gpu_mem_free": lambda x: x.free,
                "gpu_mem_used": lambda x: x.used,
                "gpu_mem_total": lambda x: x.total
            },
            "util": {
                "gpu_util": lambda x: x.gpu,
                "gpu_mem_util": lambda x: x.memory
            },
            "others": {
                "gpu_power": lambda x: nvmlDeviceGetPowerUsage(x),
                "gpu_clock_sm": lambda x: nvmlDeviceGetClockInfo(x, NVML_CLOCK_SM),
                "gpu_clock_mem": lambda x: nvmlDeviceGetClockInfo(x, NVML_CLOCK_MEM)
            }
        }
    }

    def __init__(self, enabled_metrics, connection, poll_interval=1, gpu_indices=[]):
        Process.__init__(self)
        self.exit = Event()
        self.connection = connection
        self.enabled_metrics = enabled_metrics
        self.poll_interval = poll_interval
        self.gpu_indices = gpu_indices
        self.data = []

    def run(self):
        last_collection_duration = 0
        while not self.exit.is_set():
            last_collection_duration = self.collect()
            time_to_sleep = self.poll_interval - last_collection_duration
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            if self.connection.poll():
                if self.connection.recv() == "checkpoint":
                    self.connection.send(self.data)
                    self.data = []

        self.connection.send(self.data)

    def shutdown(self):
        self.exit.set()

    def collect(self):
        start_time = time.time()
        datum = OrderedDict()
        datum["timestamp"] = start_time

        times_perc_status = [e for e in self.enabled_metrics["cpu"] if e in list(SystemMonitorCollector.metrics_collectors["cpu"]["times_perc"].keys())]
        if any(times_perc_status):
            times_perc = psutil.cpu_times_percent()
            for e in times_perc_status:
                datum[e] = SystemMonitorCollector.metrics_collectors["cpu"]["times_perc"][e](times_perc)

        for ee in [e for e in list(SystemMonitorCollector.metrics_collectors["cpu"]["others"].keys()) if e in self.enabled_metrics["cpu"]]:
            datum[ee] = SystemMonitorCollector.metrics_collectors["cpu"]["others"][ee](None)
        
        if len(self.gpu_indices) > 1:
            try:
                nvmlInit()
                for dh, idx in [(nvmlDeviceGetHandleByIndex(i), i) for i in self.gpu_indices]:

                    gpu_mem_status = [e for e in self.enabled_metrics["gpu"] if e in list(SystemMonitorCollector.metrics_collectors["gpu"]["memory"].keys())]
                    if any(gpu_mem_status):
                        gpu_mem_info = nvmlDeviceGetMemoryInfo(dh)
                        for e in gpu_mem_status:
                            datum[str(idx) + "_" + e] = SystemMonitorCollector.metrics_collectors["gpu"]["memory"][e](gpu_mem_info)
                    
                    gpu_util_status = [e for e in self.enabled_metrics["gpu"] if e in list(SystemMonitorCollector.metrics_collectors["gpu"]["util"].keys())]
                    if any(gpu_util_status):
                        gpu_util_info = nvmlDeviceGetUtilizationRates(dh)
                        for e in gpu_util_status:
                            datum[str(idx) + "_" + e] = SystemMonitorCollector.metrics_collectors["gpu"]["util"][e](gpu_util_info)
                    
                    for ee in [e for e in list(SystemMonitorCollector.metrics_collectors["gpu"]["others"].keys()) if e in self.enabled_metrics["gpu"]]:
                        datum[str(idx) + "_" + ee] = SystemMonitorCollector.metrics_collectors["gpu"]["others"][ee](dh)

                nvmlShutdown()
            except NVMLError as e:
                logger.info("NVML error: %s", e)
                pass

        end_time = time.time()
        datum["timestamp_end"] = end_time
        datum["collection_duration"] = end_time - start_time
        datum["experiment_stage"] = Experiment.current.current_stage

        self.data.append(datum)
        return end_time - start_time

class SystemMonitor(Profiler):
    _name = "system_monitor"
    possible_cpu_metrics = [
        "host_cpu_util_user", "host_cpu_util_nice", "host_cpu_util_system", 
        "host_cpu_util_idle", "host_cpu_util_iowait", "host_cpu_util_irq", 
        "host_cpu_util_softirq", "host_cpu_freq", "host_cpu_util_perc", "host_mem_prc"
    ]
    possible_gpu_metrics = [
        "gpu_mem_free", "gpu_mem_used", "gpu_mem_total", "gpu_util", 
        "gpu_mem_util", "gpu_power", "gpu_clock_sm", "gpu_clock_mem"
    ]

    def __repr__(self):
        return Profiler.__repr__(self) + ":" + SystemMonitor._name

    def __init__(self, _scope, _collection, configs=None, child_default_configs={}):
        Profiler.__init__(self, "system_monitor", "", _scope, _collection, configs=configs, child_default_configs=Configurable.merge_configs({
            "gpus": [0],
            "poll_interval": 0.5,
            "metrics": {
                "host_cpu_util_user": False,
                "host_cpu_util_nice": False,
                "host_cpu_util_system": False,
                "host_cpu_util_idle": False,
                "host_cpu_util_iowait": False,
                "host_cpu_util_irq": False,
                "host_cpu_util_softirq": False,
                "host_cpu_freq": False,
                "host_cpu_util_perc": True,
                "host_mem_prc": False,
                "gpu_mem_free": False,
                "gpu_mem_used": False,
                "gpu_mem_total": False,
                "gpu_util": True,
                "gpu_mem_util": True,
                "gpu_power": True,
                "gpu_clock_sm": False,
                "gpu_clock_mem": False
            }
        }, child_default_configs, override_first=True))

        self.enabled_metrics = {"cpu": [], "gpu": []}
        self.gpu_indices = []
    
        for pm in SystemMonitor.possible_cpu_metrics:
            if pm in self.config["metrics"] and self.config["metrics"][pm]:
                self.enabled_metrics["cpu"].append(pm)

        if "gpus" in self.config and len(self.config["gpus"]) > 0:
            for g in self.config["gpus"]:
                for pm in SystemMonitor.possible_gpu_metrics:
                    if pm in self.config["metrics"] and self.config["metrics"][pm]:
                        self.enabled_metrics["gpu"].append(pm)
                        self.gpu_indices.append(g)

    def begin(self):
        Profiler.begin(self)

    def end(self):
        self.results.clear()

    def start(self, context={}):
        key = Profiler.start(self)
        conn1, conn2 = Pipe()
        self.results[key] = {
            "ctx": OrderedDict(),
            "res": [],
            "transport_pipe": conn1,
            "collector": SystemMonitorCollector(
                self.enabled_metrics, 
                conn2, 
                poll_interval=self.config["poll_interval"], 
                gpu_indices=self.gpu_indices
            )
        }
        self.results[key]["ctx"].update(context)
        self.results[key]["collector"].start()
        return key

    def stop(self, key):
        self.results[key]["collector"].shutdown()
        res = self.results[key]["transport_pipe"].recv()
        self.results[key]["collector"].data.clear()
        self.results[key]["collector"].join()
        if "res" not in self.results[key]:
            self.results[key]["res"] = res
        else:
            self.results[key]["res"] += res
        self.results[key]["collector"] = None

    def persist(self, clear_all=True):
        outres = []
        for key, v in self.results.items():
            context = v["ctx"]
            for resdict in v["res"]:
                outdict = resdict
                outdict.update(context)
                outdict.update({"key": key})
                outres.append(outdict)
        Experiment.current.insert_many(self.collection, outres)
        if clear_all:
            self.results.clear()

