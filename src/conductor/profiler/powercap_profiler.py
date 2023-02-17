from collections import OrderedDict
from conductor.profiler.profiler import Profiler
from conductor._base import Configurable
from conductor.experiment import Experiment
from multiprocessing import Process, Event, Pipe
import time
import struct
import datetime

import logging
logger = logging.getLogger("conductor.profiler.powercap_profiler")

def read_msr(msr_index, cpu_index=0):
    try:
        cpu_msr_dev = open("/dev/cpu/%d/msr" % cpu_index, "rb")
    except IOError:
        raise Exception(("Unable to open /dev/cpu/%s/msr for reading. You need to be root." % cpu_index))
    cpu_msr_dev.seek(msr_index, 0)
    msr = cpu_msr_dev.read(8)
    msr_value = struct.unpack("<Q", msr)[0]
    cpu_msr_dev.close()
    return msr_value

def energy_delta(before, after):
    maxint = 0xffffffff
    delta = after - before
    if before > after:
        delta = after + (maxint - before)
    return delta

def time_delta(before, after):
    return (after - before)

class CPU(object):
    _name = "CPU"

    def __repr__(self):
        return CPU._name

    def __init__(self, cpu_type, cpu_id):
        if cpu_type == "Intel":
            self.cpu_type = cpu_type
            self.cpu_id = cpu_id
            self.power_unit_reg = 0x606
            self.power_pckg_reg = 0x611
            self.power_info_reg = 0x614
            self.power_unit = read_msr(self.power_unit_reg)
            power_info = read_msr(self.power_info_reg, self.cpu_id)
            tdp_unit = 1.0 / (1 << (self.power_unit & 0xF))
            self.tdp_watts = ((power_info >> 0) & 0x7FFF) * tdp_unit
        elif cpu_type == "AMD":
            self.cpu_type = cpu_type
            self.cpu_id = cpu_id
            self.power_unit_reg = 0xc0010299
            self.power_pckg_reg = 0xc001029b
            self.power_unit = read_msr(self.power_unit_reg)
            self.tdp_watts = 180
        self.energy_units = pow(0.5, (self.power_unit >> 8) & 0x1f)
        self.maxint = 0xffffffff

    def read_energy(self):
        return read_msr(self.power_pckg_reg, self.cpu_id) & self.maxint

    def get_delta(self, prior, current):
        return energy_delta(self.energy_units * prior, self.energy_units * current)

class PowercapCollector(Process):
    _name = "PowercapCollector"

    def __repr__(self):
        return PowercapCollector._name

    def __init__(self, cpus, connection, poll_interval=1):
        Process.__init__(self)
        self.exit = Event()
        self.cpus = cpus
        self.connection = connection
        self.poll_interval = poll_interval
        self.data = []
        self.dummy = {
            "cpus": [{
                "prior_energy": c.read_energy(),
                "prior_time": datetime.datetime.now()
            } for c in self.cpus]
        }

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

        for kc, c in enumerate(self.cpus):
            e_current = c.read_energy()
            e_delta_cpu = c.get_delta(self.dummy["cpus"][kc]["prior_energy"], e_current)
            self.dummy["cpus"][kc]["prior_energy"] = e_current
            
            _time = datetime.datetime.now()
            _time_delta_cpu = time_delta(self.dummy["cpus"][kc]["prior_time"], _time).total_seconds()
            self.dummy["cpus"][kc]["prior_time"] = _time

            datum["cpu" + str(kc) + "_energy"] = e_delta_cpu
            datum["cpu" + str(kc) + "_power"] = e_delta_cpu / _time_delta_cpu
            datum["cpu" + str(kc) + "_tdp"] = c.tdp_watts

        end_time = time.time()
        datum["timestamp_end"] = end_time
        datum["collection_duration"] = end_time - start_time

        self.data.append(datum)
        return end_time - start_time

class PowercapProfiler(Profiler):
    _name = "powercap_profiler"

    def __repr__(self):
        return Profiler.__repr__(self) + ":" + PowercapProfiler._name

    def __init__(self, _scope, _collection, configs=None, child_default_configs={}):
        Profiler.__init__(self, "powercap_profiler", "", _scope, _collection, configs=configs, child_default_configs=Configurable.merge_configs({
            "poll_interval": 0.5
        }, child_default_configs, override_first=True))

        socket_set = set()
        cpu_type = "Intel"
        with open("/proc/cpuinfo", 'r') as proc:
            for l in proc.readlines():
                if "vendor_id" in l:
                    if "AuthenticAMD" in l:
                        cpu_type = "AMD"
                        break
                    elif "GenuineIntel" in l:
                        cpu_type = "Intel"
                        break
        with open("/proc/cpuinfo", 'r') as proc:
            for l in proc.readlines():
                if "physical id" in l:
                    splits = l.replace(" ", "").replace("\n", "").split(":")[1]
                    socket_set.add(int(splits))

        self.cpus = []
        for cpu_id in list(socket_set):
            self.cpus.append(CPU(cpu_type, cpu_id))

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
            "collector": PowercapCollector(
                self.cpus, 
                conn2, 
                poll_interval=self.config["poll_interval"]
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
