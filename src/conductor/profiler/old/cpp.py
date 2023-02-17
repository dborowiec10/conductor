import os
import csv
import numpy as np
import uuid
from collections import OrderedDict

from conductor.core import _base

import logging
logger = logging.getLogger("conductor.profiler.cpp")

def get_avg_stddev(array):
    arr = array
    # arr = np.trim_zeros(array)
    if len(arr) > 1:
        mn = np.mean(arr)
        stddev = np.std(arr)
        return mn, stddev
    else:
        return 0, 0

class CppModelRuntimeProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_model_runtime_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()

    def init(self, header=None):
        if header:
            self.write_header(header)

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["duration"])

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                csv_writer.writerow(v["ctx"] + [v["r"]["duration"]])
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppModelPowercapProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_model_powercap_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()
        self.header_dynamic = []
        self.header_set = False

    def init(self, header=None):
        if header:
            self.header_dynamic = header + ["repeats", "slowdown_ms"]

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res
        if not self.header_set:
            for e in res["events"]:
                self.header_dynamic.append(e["name"])
            self.write_header(self.header_dynamic)
            self.header_set = True

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                data = v["ctx"] + [v["r"]["repeats"], v["r"]["slowdown_ms"]]
                for e in v["r"]["events"]:
                    data.append(e["value"])
                csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppModelNvmlProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_model_nvml_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()
        self.header_dynamic = []
        self.header_set = False

    def init(self, header=None):
        if header:
            self.header_dynamic = header + ["dev_id", "dev_name", "total_time", "num_repeats"]

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res
        if not self.header_set:
            for e in res["events"]:
                self.header_dynamic.append(e["name"] + "_avg")
                self.header_dynamic.append(e["name"] + "_stddev")
            self.write_header(self.header_dynamic)
            self.header_set = True

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                data = v["ctx"] + [v["r"]["dev_id"], v["r"]["dev_name"], v["r"]["total_time"], v["r"]["num_repeats"]]
                for e in v["r"]["events"]:
                    data.append(e["avg"])
                    data.append(e["stddev"])
                csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppPtxInstructionProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_ptx_instruction_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()

    def init(self, header=None):
        if header:
            self.write_header(header)

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["operator_id", "operator_name",  "kernel_name", "instruction", "sm_id", "exec_count"])

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                data = [] + v["ctx"]
                for k, node in enumerate(v["r"]):
                    nn = node["node_name"]
                    # for each kernel in operator
                    for kern in node["func_kernels"]:
                        kn = kern["kern_name"]
                        inst_map = {}
                        # for each block of code in each kernel
                        for block in kern["blocks"]:
                            # for each instruction in each block
                            for inst in block["instructions"]:
                                if inst not in inst_map:
                                    inst_map[inst] = []
                                for kc, c in enumerate(block["counters"]):
                                    if len(inst_map[inst]) != len(block["counters"]):
                                        inst_map[inst].append(c)
                                    else:
                                        inst_map[inst][kc] += c
                        # each instruction
                        for kinst, vinst in inst_map.items():
                            # each SM
                            for kc, vc in enumerate(vinst):
                                dat = [] + data + [k, nn, kn, kinst, kc, vc]
                                csv_writer.writerow(dat)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppScheduleRuntimeProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_schedule_runtime_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()

    def init(self, header=None):
        if header:
            self.write_header(header)

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["duration_avg", "duration_stddev", "reported_mean"])

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res
    
    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            # for each model
            for _, v in self.results.items():
                durations = v["r"]["durations"]
                mean = v["r"]["mean"]
                avg, stddev = get_avg_stddev(durations)
                data = [] + v["ctx"] + [avg, stddev, mean]
                csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppScheduleNvmlProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_schedule_nvml_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()
        self.header_dynamic = []
        self.header_set = False

    def init(self, header=None):
        if header:
            self.header_dynamic = header + ["dev_id", "dev_name", "total_time", "num_repeats"]

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res
        if not self.header_set:
            for e in res["nvml"]["events"]:
                self.header_dynamic.append(e["name"] + "_avg")
                self.header_dynamic.append(e["name"] + "_stddev")
            self.write_header(self.header_dynamic)
            self.header_set = True

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                data = v["ctx"] + [v["r"]["nvml"]["dev_id"], v["r"]["nvml"]["dev_name"], v["r"]["nvml"]["total_time"], v["r"]["nvml"]["num_repeats"]]
                for e in v["r"]["nvml"]["events"]:
                    data.append(e["avg"])
                    data.append(e["stddev"])
                csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)    
        

class CppScheduleCuptiProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_schedule_cupti_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type)
        self.results = OrderedDict()

    def init(self, header=None):
        if header:
            self.write_header(header)

    def write_header(self, header):
        if not os.path.exists(self.results_path + "_kern_details.csv") and header is not None:
            with open(self.results_path + "_kern_details.csv", "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["kernel_id", "kernel_name", "run_id", "field", "value"])
        if not os.path.exists(self.results_path + "_metrics_events.csv") and header is not None:
            with open(self.results_path + "_metrics_events.csv", "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["run_id", "num_passes", "duration", "field_type", "field_kind", "field_name", "avg_from", "value"])

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path + "_kern_details.csv", "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                kern_details = v["r"]["cupti"]["kernel_details"]
                for run_idx, run in enumerate(kern_details):
                    for kern_idx, kern in enumerate(run["kernels"]):
                        for kern_key, kern_val in sorted(kern.items()):
                            if kern_key != "name":
                                data = [] + v["ctx"] + [kern_idx, kern["name"], run_idx, kern_key, kern_val]
                                csv_writer.writerow(data)

        with open(self.results_path + "_metrics_events.csv", "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                durations = v["r"]["cupti"]["durations"]
                num_passes = v["r"]["cupti"]["num_passes"]                
                metrics_events = v["r"]["cupti"]["metrics"]
                for run_idx, run in enumerate(metrics_events):
                    for met in run["metrics"]:
                        data = [] + v["ctx"] + [run_idx, num_passes[run_idx], durations[run_idx], "metric", met["kind"], met["name"], met["avg_from"], met["value"]]
                        csv_writer.writerow(data)
                    for ev in run["events"]:
                        data = [] + v["ctx"] + [run_idx, num_passes[run_idx], durations[run_idx], "event", ev["kind"], ev["name"], ev["avg_from"], ev["value"]]
                        csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppOperatorRuntimeProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_operator_runtime_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()

    def init(self, header=None):
        if header:
            self.write_header(header)

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["operator_id",  "operator_name", "duration_avg", "duration_stddev"])

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                for op in v["r"]:
                    avg, stddev = get_avg_stddev(op["durations"])
                    data = [] + v["ctx"] + [op["id"], op["name"], avg, stddev]
                    csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppTuningNvmlProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_tuning_nvml_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()
        self.header_dynamic = []
        self.header_set = False
        # self.lock = threading.Lock()

    def init(self, header=None):
        if header:
            self.header_dynamic = header + ["dev_id", "dev_name", "total_time", "num_repeats"]

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def append_to_context(self, key, ctx):
        self.results[key]["ctx"] = self.results[key]["ctx"] + ctx

    def set_results(self, key, res):
        if res:
            self.results[key]["r"] = res
            if not self.header_set:
                for e in res["events"]:
                    self.header_dynamic.append(e["name"] + "_avg")
                    self.header_dynamic.append(e["name"] + "_stddev")
                self.write_header(self.header_dynamic)
                self.header_set = True

    def persist(self, header=None, clear=False):
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                if v["r"]:
                    data = v["ctx"] + [v["r"]["dev_id"], v["r"]["dev_name"], v["r"]["total_time"], v["r"]["num_repeats"]]
                    for e in v["r"]["events"]:
                        data.append(e["avg"])
                        data.append(e["stddev"])
                    csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppOperatorNvmlProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_operator_nvml_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type + ".csv")
        self.results = OrderedDict()
        self.header_dynamic = []
        self.header_set = False

    def init(self, header=None):
        if header:
            self.header_dynamic = header + ["operator_id", "operator_name", "dev_id", "dev_name", "total_time", "num_repeats", "duration_avg", "duration_stddev"]

    def write_header(self, header):
        if not os.path.exists(self.results_path) and header is not None:
            with open(self.results_path, "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res
        if not self.header_set:
            for e in res[0]["nvml_report"]["events"]:
                self.header_dynamic.append(e["name"] + "_avg")
                self.header_dynamic.append(e["name"] + "_stddev")
            self.write_header(self.header_dynamic)
            self.header_set = True

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path, "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                for op in v["r"]:
                    dur_avg, dur_stddev = get_avg_stddev(op["durations"])
                    avg_report = op["nvml_report"]
                    data = [] + v["ctx"] + [op["id"], op["name"], avg_report["dev_id"], avg_report["dev_name"], avg_report["total_time"], avg_report["num_repeats"], dur_avg, dur_stddev]
                    for e in avg_report["events"]:
                        data.append(e["avg"])
                        data.append(e["stddev"])
                    csv_writer.writerow(data)
        if clear:
            self.results.clear()

    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)


class CppOperatorCuptiProfiler(_base.Profiler):
    configuration_slots = ["general"]

    def __init__(self, _id, configs, scope, results_path):
        _base.Profiler.__init__(self, _id, "cpp_operator_cupti_profiler", configs, scope, results_path)
        self.results_path = os.path.join(self.results_path, self._scope + "_" + self._type)
        self.results = OrderedDict()

    def init(self, header=None):
        if header:
            self.write_header(header)

    def write_header(self, header):
        if not os.path.exists(self.results_path + "_kern_details.csv") and header is not None:
            with open(self.results_path + "_kern_details.csv", "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["operator_id", "operator_name", "kernel_id", "kernel_name", "run_id", "field", "value"])
        if not os.path.exists(self.results_path + "_metrics_events.csv") and header is not None:
            with open(self.results_path + "_metrics_events.csv", "a+") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header + ["operator_id", "operator_name", "run_id", "num_passes", "duration", "field_type", "field_kind", "field_name", "avg_from", "value"])

    def finish(self):
        self.results.clear()

    def start(self, context):
        key = str(uuid.uuid4())
        self.results[key] = {
            "ctx": context,
            "r": None
        }
        return key

    def stop(self, key):
        pass

    def set_results(self, key, res):
        self.results[key]["r"] = res

    def persist(self, header=None, clear=False):
        self.write_header(header)
        with open(self.results_path + "_kern_details.csv", "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                for op in v["r"]:
                    op_id = op["id"]
                    op_name = op["name"]
                    kern_details = op["kernel_details"]
                    for run_idx, run in enumerate(kern_details):
                        for kern_idx, kern in enumerate(run["kernels"]):
                            for kern_key, kern_val in sorted(kern.items()):
                                if kern_key != "name":
                                    data = [] + v["ctx"] + [op_id, op_name, kern_idx, kern["name"], run_idx, kern_key, kern_val]
                                    csv_writer.writerow(data)

        with open(self.results_path + "_metrics_events.csv", "a+") as csv_file:
            csv_writer = csv.writer(csv_file)
            for _, v in self.results.items():
                for op in v["r"]:
                    op_id = op["id"]
                    op_name = op["name"]
                    durations = op["durations"]
                    num_passes = op["num_passes"]                
                    metrics_events = op["metrics"]
                    for run_idx, run in enumerate(metrics_events):
                        for met in run["metrics"]:
                            data = [] + v["ctx"] + [op_id, op_name, run_idx, num_passes[run_idx], durations[run_idx], "metric", met["kind"], met["name"], met["avg_from"], met["value"]]
                            csv_writer.writerow(data)
                        for ev in run["events"]:
                            data = [] + v["ctx"] + [op_id, op_name, run_idx, num_passes[run_idx], durations[run_idx], "event", ev["kind"], ev["name"], ev["avg_from"], ev["value"]]
                            csv_writer.writerow(data)
        if clear:
            self.results.clear()
            
    @classmethod
    def create(cls, _id, configs, scope, results_path):
        return cls(_id, configs, scope, results_path)
