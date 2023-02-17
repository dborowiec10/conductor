
# This file defines mediating overrides of tvm functions that enable more control over task extraction, application of configurations to tensor programs and so on
# It acts as a binding layer between Conductor's API and TVM's, AutoTVM's and AutoScheduler's low level APIs

from conductor.utils import get_const_tuple, generic_topi_unwrapper
from conductor.component.method.flex.task import Task as FlexTask, TASK_TABLE as FLEX_TASK_TABLE, register_compute as flex_register_compute
from conductor.component.method.flex.utils import Config as FlexConfig
from conductor.tensor_program import TensorProgram
from tvm.autotvm.tophub import _get_tophub_location, AUTOTVM_TOPHUB_NONE_LOC, PACKAGE_VERSION, _alias, AUTOTVM_TOPHUB_ROOT_PATH, check_backend

import tvm
from tvm.tir.expr import FloatImm, IntImm, StringImm, Var, Any
from tvm.ir import container as tvm_container
from tvm.runtime import container as tvm_runtime_container
from tvm.te import tensor, placeholder
from tvm import __version__
from tvm.target import Target
from tvm import IRModule
from tvm.ir.transform import PassContext
from tvm.relay.function import Function
from tvm._ffi import register_func

from tvm.auto_scheduler.relay_integration import TracingEnvironment, TracingMode, traverse_to_get_io_tensors
from tvm.auto_scheduler.workload_registry import serialize_workload_registry_entry, deserialize_workload_registry_entry, workload_key_to_tensors, register_workload_tensors, WORKLOAD_FUNC_REGISTRY as SKETCH_FUNC_REGISTRY
from tvm.auto_scheduler.utils import decode_workload_key
from tvm.auto_scheduler.compute_dag import ComputeDAG, LayoutRewriteOption
from tvm.auto_scheduler.dispatcher import DispatchContext as AnsorDispatchContext
from tvm.auto_scheduler.measure import prepare_input_map, recover_measure_input
from tvm.auto_scheduler import SearchTask as SketchTask, register_workload
from tvm.auto_scheduler._ffi_api import SerializeMeasureInput, DeserializeMeasureInput, WriteMeasureRecords, ReadMeasureRecord
from tvm.autotvm.task.space import FallbackConfigEntity, ConfigEntity, ConfigSpace
from tvm.autotvm.task.dispatcher import DispatchContext as AutoTVMDispatchContext
from tvm.autotvm.task.topi_integration import TaskExtractEnv, register_topi_compute, register_topi_schedule
from tvm.autotvm.task.task import compute_flop as template_compute_flop, create as autotvm_task_create, Task as TemplateTask, TASK_TABLE as TEMPLATE_TASK_TABLE, template as create_template
from tvm.autotvm import GLOBAL_SCOPE


from collections import namedtuple, OrderedDict
import undecorated
import numpy as np
import json
import os

import logging
logger = logging.getLogger("conductor.mediation")

try:  # convert unicode to str for python2
    _unicode = unicode
except NameError:
    _unicode = ()

try:
    _long = long
except NameError:
    _long = int


ERROR_TYPES = [
    "SUCCESS",
    "INSTANTIATION_ERROR",
    "COMPILE_HOST_ERROR",
    "COMPILE_DEVICE_ERROR",
    "RUNTIME_DEVICE_ERROR",
    "WRONG_ANSWER_ERROR",
    "BUILD_TIMEOUT_ERROR",
    "RUN_TIMEOUT_ERROR",
    "UNKNOWN_ERROR",
    "SKIP"
]

class MeasureErrorNo(object):
    """ 
    Error type for MeasureResult. 
    """
    _name = "measurer_error_no"
    NO_ERROR = 0  # No error
    INSTANTIATION_ERROR = 1  # Errors happen when apply transform steps from init state
    COMPILE_HOST = 2  # Errors happen when compiling code on host (e.g., tvm.build)
    COMPILE_DEVICE = 3  # Errors happen when compiling code on device (e.g. OpenCL JIT on the device)
    RUNTIME_DEVICE = 4  # Errors happen when run program on device
    WRONG_ANSWER = 5  # Answer is wrong when compared to a reference output
    BUILD_TIMEOUT = 6  # Timeout during compilation
    RUN_TIMEOUT = 7  # Timeout during run
    UNKNOWN_ERROR = 8  # Unknown error
    SKIP = 9 # Skip

    def __repr__(self):
        return MeasureErrorNo._name

class MeasureInput(object):
    """
    Stores all the necessary inputs for a measurement.
    """
    _name = "measure_input"

    def __repr__(self):
        return MeasureInput._name

    def __init__(self, task, workload_key, target, config, theoretical_flop):
        self.task = task
        self.workload_key = workload_key
        self.target = target
        self.config = config
        self.theoretical_flop = theoretical_flop

    # dummy method to conform to ansor api so that measurer/runner api can be unified and independent
    def serialize(self):
        return [
            SerializeMeasureInput(self),
            serialize_workload_registry_entry(self.task.workload_key),
        ]

    def unique_key(self):
        if isinstance(self.task, SketchTask):
            return SerializeMeasureInput(self.config)
        else:
            return "" + str(self.target) + str(self.workload_key) + str(self.task.args) + str(self.config)

    # dummy method to conform to ansor api so that measurer/runner api can be unified and independent
    @staticmethod
    def deserialize(data):
        inp = DeserializeMeasureInput(data[0])
        deserialize_workload_registry_entry(data[1])
        return recover_measure_input(inp)

    @staticmethod
    def from_sketch_input(inp):
        # in cases where measure input is constructed but recover_measure_input is not called:
            # i.e. when decoding logfile for model
            # at that point, tasks will not be registered for ansor since its all happening internally
            # thus, we just need create the measure input
        if inp.task.compute_dag is not None:
            flp = inp.task.compute_dag.get_flop()
        else:
            flp = 0
        return MeasureInput(inp.task, inp.task.workload_key, inp.task.target, inp.state, flp)

class BuildInput(namedtuple("BuildInput", ["target", "target_host", "schedule", "args", "error_no", "error_msg", "options"])):
    """
    N/A
    """
    def __repr__(self):
        return ":build_input"

class BuildResult(namedtuple("BuildResult", ("filename", "args", "error_no", "error_msg", "time_cost", "timestamp", "status"))):
    """
    N/A
    """
    def __repr__(self):
        return ":build_result"

class ProfileResult(namedtuple("ProfileResult", ["mean", "results", "total_time", "other"])):
    """
    N/A
    """
    def __repr__(self):
        return ":profile_result"

class MeasureResult(object):
    _name = "measure_result"

    def __repr__(self):
        return MeasureResult._name

    def __init__(self, costs, error_no, error_msg, all_cost, timestamp, achieved_flop, mean, total_time, status, other=None):
        self.costs = costs
        self.error_no = error_no
        self.error_msg = error_msg
        self.all_cost = all_cost
        self.timestamp = timestamp
        self.achieved_flop = achieved_flop
        self.mean = mean
        self.total_time = total_time
        self.status = status
        self.other = other

def serialize_args(args):
    def _encode(x):
        if isinstance(x, tensor.Tensor):
            return ("TENSOR", get_const_tuple(x.shape), x.dtype)
        if isinstance(x, (tuple, list, tvm_container.Array)):
            return tuple([_encode(a) for a in x])
        if isinstance(x, (str, int, float, Var, Any)):
            return x
        if isinstance(x, (StringImm, IntImm, FloatImm)):
            return x.value
        if isinstance(x, tvm_runtime_container.String):
            return str(x)
        if x is None:
            return None
        raise RuntimeError(
            'Do not support type "%s" in argument. Consider to use'
            "primitive types or tvm.tir.Var only" % type(x)
        )
    ret = []
    for t in args:
        ret.append(_encode(t))
    return tuple(ret)


def deserialize_args(args):
    """The inverse function of :code:`serialize_args`.

    Parameters
    ----------
    args: list of hashable or Tensor
    """
    ret = []
    for t in args:
        if isinstance(t, tuple) and t[0] == 'TENSOR':
            ret.append(placeholder(shape=t[1], dtype=t[2]))
        else:
            ret.append(t)
    return ret

def extract_inputs(out):
    inputs = []
    queue = [out]
    hash_set = set()
    while queue:
        t = queue.pop(0)
        if isinstance(t.op, tensor.PlaceholderOp):
            inputs.append(t)
        else:
            input_tensors = [t for t in t.op.input_tensors if t not in hash_set]
            queue.extend(input_tensors)
            hash_set.update(input_tensors)
    return inputs

def sketch_wrapper(func):
    def wrapper(*args, **kwargs):
        if callable(func):
            out = func(*args, **kwargs)
        else:
            out = func
    
        if isinstance(out, (list, tuple, tvm.container.Array)):
            ret_out = out
        else:
            ret_out = [out]

        output_tensors = []
        for i in ret_out:
            if not isinstance(i.op, tensor.PlaceholderOp):
                output_tensors.append(i)
        _input_tensors = set()
        for o in output_tensors:
            for ii in extract_inputs(o):
                _input_tensors.add(ii)
        ret = list(_input_tensors) + output_tensors
        return ret

    setattr(wrapper, 'func_name', getattr(func, "func_name"))
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    return wrapper

def _get_inputs(out):
    inputs = []
    outputs = []
    queue = [out] if not isinstance(out, (list, tuple)) else out
    hash_set = set()
    
    while queue:
        t = queue.pop(0)
        if isinstance(t.op, tvm.te.tensor.PlaceholderOp):
            inputs.append(t)
        else:
            outputs.append(t.op)
            input_tensors = [t for t in t.op.input_tensors if t not in hash_set]
            queue.extend(input_tensors)
            hash_set.update(input_tensors)

    return inputs, outputs

def flex_wrapper(func):
    def wrapper(*args, **kwargs):
        if callable(func):
            out = func(*args, **kwargs)
        else:
            out = func
        ts = out if isinstance(out, (list, tuple)) else [out]
        inps, outps = _get_inputs(ts[-1:])
        return (outps, inps)
    setattr(wrapper, 'func_name', getattr(func, "func_name"))
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    return wrapper

def workload_key_from_log(dec_line, _type):
    if _type == "sketch":
        return dec_line["i"][0][0]
    else:
        return dec_line["input"][1]

def identifier_from_log(key, _type):
    if _type == "sketch":
        return json.loads(key)[0]
    elif _type == "template":
        return key[0]
    elif _type == "flex":

        key_splits = key.split(":")
        name = key_splits[1]
        return name
    else:
        return None

def determine_log_type(_str):
    j = json.loads(_str)
    if "i" in j and "r" in j:
        wkld_key = workload_key_from_log(j, "sketch")
        return (j, "sketch", wkld_key, identifier_from_log(wkld_key, "sketch"))
    elif "input" in j and "config" in j:
        c = j["config"]
        if "op_config_lst" in c and "graph_config" in c:
            wkld_key = workload_key_from_log(j, "flex")
            return (j, "flex", wkld_key, identifier_from_log(wkld_key, "flex"))
        else:
            wkld_key = workload_key_from_log(j, "template")
            return (j, "template", wkld_key, identifier_from_log(wkld_key, "template"))
    else:
        return (j, None, None, None)

def encode_measure_input_result(inp, res, k=0):
    if isinstance(inp.task, SketchTask):
        _str = WriteMeasureRecords(inp.config, res.other)
        inp_dict = json.loads(_str)
        return (_str, {
            "idx": k,
            "input": json.dumps(inp_dict["i"][0]),
            "config": json.dumps(inp_dict["i"][1]),
            "result": (
                [float(f.value) if isinstance(f, FloatImm) else f for f in res.costs] if res.error_no == 0 else (1e9,),
                int(res.error_no),
                float(res.all_cost),
                float(res.timestamp),
                float(res.achieved_flop)
            ),
            "version": 0.2,
            "tvm_version": __version__,
        })
    else:
        if isinstance(inp.task, FlexTask):
            conf = inp.config.serialize()
        else:
            conf = inp.config.to_json_dict()

        ser_args = serialize_args(inp.task.args)
        json_dict = {
            "input": (str(inp.target), inp.workload_key, inp.theoretical_flop, ser_args),
            "config": conf,
            "result": (
                [float(f.value) if isinstance(f, FloatImm) else f for f in res.costs] if res.error_no == 0 else (1e9,),
                int(res.error_no),
                float(res.all_cost),
                float(res.timestamp),
                float(res.achieved_flop)
            ),
            "version": 0.2,
            "tvm_version": __version__,
        }
        _str = json.dumps(json_dict)
        json_dict["idx"] = k
        return (_str, json_dict)

def decode_measure_input_result(_str):
    if _str and not _str.startswith("#") and _str != "" and _str != " ":
        decoded_line, log_type, _key, identifier = determine_log_type(_str)

        if log_type == "sketch":
            if not Tasker.check_registered_compute(_key, log_type):
                # check if we have a tensor program for this
                found = False
                try:
                    tp = TensorProgram(identifier, None, None, None, None, None, None, load=True) # this will autoload it from main repository
                    if tp.is_topi:
                        compute = generic_topi_unwrapper(tp.compute_gen_func, tp.target, tp.func_name)
                    else:
                        compute = tp.compute_gen_func
                    compute = sketch_wrapper(compute)
                    Tasker.register_compute(tp.func_name, compute, tp.comp_sched_gen_func, "sketch")
                    found = True
                except Exception as e:
                    pass
            
            inp, result = ReadMeasureRecord(_str)
            if found:
                inp = recover_measure_input(inp, rebuild_state=True)
            inp = MeasureInput.from_sketch_input(inp)

        else:
            costs, error_no, all_cost, timestamp, achieved_flop = decoded_line["result"]
            result = MeasureResult(costs, error_no, "", all_cost, timestamp, achieved_flop, np.mean(costs), all_cost, "*")
            target, wkld_key, theoretical_flop, targs, _ = decoded_line["input"]
            target = str(target)
            if "-target" in target:
                target = target.replace("-target", "-mtriple")
            target = Target(target)
            if log_type == "template":
                def clean_json_to_python(x):
                    if isinstance(x, list):
                        return tuple([clean_json_to_python(a) for a in x])
                    if isinstance(x, _unicode):
                        return str(x)
                    if isinstance(x, (_long, int)):
                        return int(x)
                    return x
                cleaned_key = clean_json_to_python(wkld_key)
                cleaned_args = clean_json_to_python(targs)

                if not Tasker.check_registered_compute(cleaned_key[0], "template"):
                    tp = TensorProgram(identifier, None, None, None, None, None, None, load=True) # this will autoload it from main repository
                    Tasker.register_compute(cleaned_key[0], tp.compute_gen_func, tp.comp_sched_gen_func, "template")
                
                temp_task = autotvm_task_create(cleaned_key[0], cleaned_args, target)
                config = ConfigEntity.from_json_dict(decoded_line["config"])
                config.cost = np.mean(costs)
                inp = MeasureInput(temp_task, temp_task.workload, target, config, theoretical_flop)

            elif log_type == "flex":
                key_splits = wkld_key.split(":")
                category = key_splits[0]
                name = key_splits[1]

                if not Tasker.check_registered_compute(wkld_key, log_type):
                    tp = TensorProgram(identifier, None, None, None, None, None, None, load=True) # this will autoload it from main repository
                    if tp.is_topi:
                        comp_def = generic_topi_unwrapper(tp.compute_gen_func, tp.target, tp.func_name)
                    else:
                        comp_def = tp.compute_gen_func
                    compute = flex_wrapper(comp_def)
                    temp_task = FlexTask(tp.func_name, tp.func_name, compute, tp.args, target, tp.target_host)
                    Tasker.register_compute(temp_task.key, temp_task, None, "flex")
                else:
                    temp_task = FlexTask(category, name, None, deserialize_args(targs), target, None)

                config = FlexConfig.create(json.loads(decoded_line["config"]))
                inp = MeasureInput(temp_task, wkld_key, target, config, theoretical_flop)
            else:
                raise RuntimeError("Invalid log entry!")
        return (inp, result), log_type

class DispatchContext(object):
    """
    Class representing the dispatch context
    Stores information about configurations for each operator during schedule creation
    
    Attributes
    ----------
    """

    _name = "dispatch_context"
    current = None

    def __repr__(self):
        return DispatchContext._name

    def __init__(self):
        self._old_ctx = DispatchContext.current
        self.old = DispatchContext.current
        self.old_template = AutoTVMDispatchContext.current
        self.old_sketch = AnsorDispatchContext.current
        self.scheduler = None

    def isfallback(self):
        """
        Determines whether the context is a fallback context

        Returns
        -------
        bool
            Whether the context is fallback
        """
        return False

    def __enter__(self):
        self.old = DispatchContext.current
        self._old_ctx = DispatchContext.current
        DispatchContext.current = self
        self.old_template = AutoTVMDispatchContext.current
        self.old_sketch = AnsorDispatchContext.current
        AutoTVMDispatchContext.current = self
        AnsorDispatchContext.current = self
        return self

    def register_scheduler(self, scheduler):
        self.scheduler = scheduler

    # Generate a schedule using a registered scheduler function
    # This is used during compilation to apply configs to tensor programs
    # Applied configs should generate schedules which then can be compiled to code
    def schedule_schedule(self, func_name, coded_key, decoded_key, target, io_tensors, args, dag, input_map, config, has_layout_free, has_complex_op):
        return self.scheduler.from_autoschedule_topi(func_name, coded_key, decoded_key, target, io_tensors, args, dag, input_map, config, has_layout_free, has_complex_op)
    
    def __exit__(self, ptype, value, trace):
        DispatchContext.current = self.old
        AutoTVMDispatchContext.current = self.old_template
        AnsorDispatchContext.current = self.old_sketch

    def query(self, target, key):
        ret = self._query_inside(target, key)
        if ret is None:
            ret = self.old.query(target, key)
        return ret

class FallbackContext(DispatchContext):
    _name = "fallback"

    def __repr__(self):
        return DispatchContext.__repr__(self) + ":" + FallbackContext._name

    def __init__(self):
        DispatchContext.__init__(self)
        self.memory = {}

    def isfallback(self):
        return True

    def clear_cache(self, target, key):
        k = (str(target), key)
        if k in self.memory:
            del self.memory[k]

    def update(self, target, key, cfg):
        k = (str(target), key)
        self.memory[k] = cfg

    def query(self, target, key):
        return self._query_inside(target, key)

    def _query_inside(self, target, key):
        k = (str(target), key)
        if k in self.memory:
            return self.memory[k]
        self.memory[k] = FallbackConfigEntity()
        return self.memory[k]
    
class SingleConfigContext(DispatchContext):
    _name = "single_config"

    def __repr__(self):
        return DispatchContext.__repr__(self) + ":" + SingleConfigContext._name

    def __init__(self, config):
        DispatchContext.__init__(self)
        self._config = config

    def _query_inside(self, target, key):
        return self._config

    def update(self, target, key, config):
        self._config = config

DispatchContext.current = FallbackContext()

class HistoryBestContext(DispatchContext):
    _name = "history_best"
    
    def __repr__(self):
        return DispatchContext.__repr__(self) + ":" + HistoryBestContext._name

    def __init__(self, records):
        DispatchContext.__init__(self)
        self.best_by_key = {}
        self.best_by_model = {}
        self.best_user_defined = {}
        self.load(records)
        self.record_type = None
    
    def load_records(self, fname):
        records = []
        with open(fname, "r") as _fil:
            for row in _fil:
                ret, record_type = decode_measure_input_result(row)
                self.record_type = record_type
                if ret is None:
                    continue
                else:
                    records.append(ret)
        return records

    def load(self, recs):
        if isinstance(recs, list):
            records = recs
        elif isinstance(recs, str):
            records = self.load_records(recs)
        counter = 0
        for inp, res in records:
            counter += 1
            if res.error_no != MeasureErrorNo.NO_ERROR:
                continue
            for kk in inp.target.keys:
                kki = (kk, inp.workload_key)
                if kki not in self.best_by_key:
                    self.best_by_key[kki] = (inp, res)
                else:
                    _, other_res = self.best_by_key[kki]
                    other_costs = [x.value if isinstance(x, FloatImm) else x for x in other_res.costs]
                    costs = [x.value if isinstance(x, FloatImm) else x for x in res.costs]
                    if np.mean(other_costs) > np.mean(costs):
                        self.best_by_key[kki] = (inp, res)

            kki = (inp.target.model, inp.workload_key)
            if kki not in self.best_by_model:
                if inp.target.model != "unknown":
                    self.best_by_model[kki] = (inp, res)
            else:
                _, other_res = self.best_by_model[kki]
                other_costs = [x.value if isinstance(x, FloatImm) else x for x in other_res.costs]
                costs = [x.value if isinstance(x, FloatImm) else x for x in res.costs]
                if np.mean(other_costs) > np.mean(costs):
                    self.best_by_model[kki] = (inp, res)

    def _query_inside(self, target, key):
        if target is None:
            raise RuntimeError("Need a target context to find the history best")
        k = (target.model, key)
        if k in self.best_user_defined:
            return self.best_user_defined[k]
        if k in self.best_by_model:
            inp, _ = self.best_by_model[k]
            return inp.config
        for kk in target.keys:
            kki = (kk, key)
            if kki in self.best_user_defined:
                return self.best_user_defined[kki]
            if kki in self.best_by_key:
                inp, _ = self.best_by_key[kki]
                return inp.config
        return None

    def update(self, target, key, config):
        model = target.model
        k = (model, key)
        self.best_user_defined[k] = config
        for kk in target.keys:
            kki = (kk, key)
            self.best_user_defined[kki] = config

class TophubContext(object):
    current = None
    _name = "dispatch_context:tophub"
    
    def __repr__(self):
        return TophubContext._name

    def _decode(self, string_row):
        decoded_line, log_type, _key, identifier = determine_log_type(string_row)
        target, wkld_key, targs, _ = decoded_line["input"]
        target = str(target)
        if "-target" in target:
            target = target.replace("-target", "-mtriple")
        target = Target(target)
        def clean_json_to_python(x):
            if isinstance(x, list):
                return tuple([clean_json_to_python(a) for a in x])
            if isinstance(x, _unicode):
                return str(x)
            if isinstance(x, (_long, int)):
                return int(x)
            return x
        cleaned_key = clean_json_to_python(wkld_key)
        cleaned_args = clean_json_to_python(targs)
        if not Tasker.check_registered_compute(cleaned_key, "template"):
            tp = TensorProgram(identifier, None, None, None, None, None, None, load=True) # this will autoload it from main repository
            Tasker.register_compute(cleaned_key, tp.compute_gen_func, tp.comp_sched_gen_func, "template")
        try:
            temp_task = autotvm_task_create(cleaned_key, cleaned_args, target)
        except Exception as e:
            return None
        
        theoretical_flop = Tasker.task_theoretical_flop(temp_task)
        config = ConfigEntity.from_json_dict(decoded_line["config"])
        inp = MeasureInput(temp_task, wkld_key, target, config, theoretical_flop)
        costs, error_no, all_cost, timestamp = decoded_line["result"]
        achieved_flop = theoretical_flop / np.mean(costs)
        result = MeasureResult(costs, error_no, "", all_cost, timestamp, achieved_flop, np.mean(costs), all_cost, "*")
        return (inp, result)

    def __init__(self, target):
        tophub_location = _get_tophub_location()
        if tophub_location == AUTOTVM_TOPHUB_NONE_LOC:
            self.ctx = FallbackContext()
        else:
            records = []
            targets = target if isinstance(target, (list, tuple)) else [target]
            for tgt in targets:
                possible_names = []
                device = tgt.attrs.get("device", "")
                if device != "":
                    possible_names.append(_alias(device))
                possible_names.append(tgt.kind.name)
                all_packages = list(PACKAGE_VERSION.keys())
                for name in possible_names:
                    name = _alias(name)
                    if name in all_packages:
                        if not check_backend(tophub_location, name):
                            continue
                        filename = "%s_%s.log" % (name, PACKAGE_VERSION[name])
                        for row in open(os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, filename)):
                            if row and not row.startswith("#") and row != "":
                                ret = self._decode(row)
                                if ret is None:
                                    continue
                                else:
                                    records.append(ret)
                        break
            
            self.ctx = HistoryBestContext(records)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "scheduler":
            self.ctx.__setattr__(__name, __value)
        else:
            object.__setattr__(self, __name, __value)
    
    def __getattribute__(self, __name: str) -> Any:
        if __name == "scheduler":
            return self.ctx.__getattribute__(__name)
        else:
            return super().__getattribute__(__name)

    def isfallback(self):
        return self.ctx.isfallback()

    def __enter__(self):
        return self.ctx.__enter__()

    def __exit__(self, ptype, value, trace):
        return self.ctx.__exit__(ptype, value, trace)
    
    def register_scheduler(self, _func):
        self.ctx.register_scheduler(_func)

    def schedule_schedule(self, func_name, coded_key, decoded_key, target, io_tensors, compute_def, args, input_map, config, has_layout_free, has_complex_op):
        return self.ctx.schedule_schedule(func_name, coded_key, decoded_key, target, io_tensors, compute_def, args, input_map, config, has_layout_free, has_complex_op)

    def query(self, target, key):
        return self.ctx.query(target, key)

    def update(self, target, key, config):
        return self.ctx.update(target, key, config)

    def _query_inside(self, target, key):
        return self.ctx._query_inside(target, key)

    def load(self, records):
        return self.ctx.load(records)

class TensorProgramExtractionEnvironment(object):
    _name = "tensor_program_extraction_environment"
    current = None

    def __repr__(self):
        return TensorProgramExtractionEnvironment._name

    def __init__(self, tracing_mode=TracingMode.EXTRACT_TASK):
        self.old = TensorProgramExtractionEnvironment.current
        self.old_template = TaskExtractEnv.current
        self.old_sketch = TracingEnvironment.current
        self.tracing_mode = tracing_mode
        self.tracing = False
        self.wanted_relay_ops = None
        self.tensor_program_definitions = OrderedDict()
        self.func_name_to_key = {}


    def __enter__(self):
        self.old = TensorProgramExtractionEnvironment.current
        self.old_template = TaskExtractEnv.current
        self.old_sketch = TracingEnvironment.current
        TensorProgramExtractionEnvironment.current = self
        TaskExtractEnv.current = self
        TracingEnvironment.current = self
        self.tracing = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TensorProgramExtractionEnvironment.current = self.old
        TaskExtractEnv.current = self.old_template
        TracingEnvironment.current = self.old_sketch
        self.tracing = False

    def reset(self):
        pass

    # required for TOPI integration of autotvm
    def add_task(self, name, args, node, outs, target):
        templ = TEMPLATE_TASK_TABLE[name]

        if name + str(args) in self.tensor_program_definitions:
            self.tensor_program_definitions[name + str(args)]["weight"] += 1
        else:
            self.tensor_program_definitions[name + str(args)] = {
                "func_names": set(),
                "workload_key": name + str(args),
                "target": target,
                "io_tensors": outs,
                "compute_gen_func": undecorated.undecorated(templ.fcompute),
                "schedule_gen_func": undecorated.undecorated(templ.fschedule),
                "args": args,
                "input_map": None,
                "has_layout_free": True,
                "has_complex_op": True,
                "weight": 1
            }
            self.tensor_program_definitions[name + str(args)]["func_names"].add(name)
        self.func_name_to_key[name] = name + str(args)

    def add_tensor_program(self, func_name, coded_key, target, io_tensors, args, input_map, has_layout_free, has_complex_op):
        if coded_key in self.tensor_program_definitions:
            self.tensor_program_definitions[coded_key]["weight"] += 1
            self.tensor_program_definitions[coded_key]["func_names"].add(func_name)
        else:
            self.tensor_program_definitions[coded_key] = {
                "func_names": set(),
                "workload_key": coded_key,
                "target": target,
                "io_tensors": io_tensors,
                "compute_gen_func": None,
                "schedule_gen_func": None,
                "args": args,
                "input_map": input_map,
                "has_layout_free": has_layout_free,
                "has_complex_op": has_complex_op,
                "weight": 0
            }
            self.tensor_program_definitions[coded_key]["func_names"].add(func_name)
        self.func_name_to_key[func_name] = coded_key

    def get_tensor_programs(self):
        return self.tensor_program_definitions

# Traces the module to extract tensor programs
# Amalgamation of both AutoTVM and AutoScheduler code
def trace_module(module, params, target, tracing_env, opt_level=3):
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    # Turn off AutoTVM config not found warnings
    old_autotvm_silent = GLOBAL_SCOPE.silent
    GLOBAL_SCOPE.silent = True

    if isinstance(target, str):
        _target = Target(target)
    else:
        _target = target

    # create target
    if isinstance(module, relay.function.Function):
        mod = IRModule.from_expr(module)
    else:
        mod = module
    assert isinstance(mod, IRModule), "only support relay Module or Function to be tuned"

    with tracing_env:
        with PassContext(
            opt_level=opt_level, 
            config={"relay.backend.use_auto_scheduler": True}, 
            disabled_pass={"AutoSchedulerLayoutRewrite"}):
            compiler = relay.vm.VMCompiler()
            if params:
                compiler.set_params(params)
            mod = IRModule.from_expr(mod) if isinstance(mod, relay.Function) else mod
            compiler.lower(mod, _target)

    GLOBAL_SCOPE.silent = old_autotvm_silent

hack_map = [
    ("resnet18_v1", None), 
    ("alexnet", "winograd"),
    ("mobilenet", "winograd"),
    ("squeezenet", "winograd"),
    ("vgg", "winograd")]

def get_from_hack_map(model_name):
    for m, disabled in hack_map:
        if m in model_name:
            return disabled
    return "#"

class Tasker(object):
    _name = "tasker"

    def __repr__(self):
        return Tasker._name
    
    @staticmethod
    def extract_tensor_programs(module, params, model_name, target, target_host):
        if isinstance(target, str):
            target = Target(target)
        if isinstance(target_host, str):
            target_host = Target(target_host)
        if isinstance(module, Function):
            module = IRModule.from_expr(module)
        assert isinstance(module, IRModule), "only support relay Module or Function to be tuned"

        tpenv = TensorProgramExtractionEnvironment()
        trace_module(module, params, target, tpenv)
        defs = tpenv.tensor_program_definitions

        templateable = []
        untemplateable = []
        for k, d in defs.items():
            if d["schedule_gen_func"] is not None:
                templateable.append((k, d))
            else:
                untemplateable.append((k, d))

        tensor_programs = []
        for idx, (k, d) in enumerate(templateable):
            identifier = model_name + "_" + str(idx) + "_" + k
            _tp = TensorProgram(
                identifier,
                k,
                d["compute_gen_func"],
                d["schedule_gen_func"],
                d["args"],
                target,
                target_host,
                is_topi=True,
                io_tensors=d["io_tensors"]
            )
            # tp = {
            #     "key": k,
            #     "def": d,
            #     "kind": "templateable",
            #     "tasks": {
            #         "template": Tasker.task_from_tensor_program(_tp, "template"),
            #         "sketch": Tasker.task_from_tensor_program(_tp, "sketch"),
            #         "flex": Tasker.task_from_tensor_program(_tp, "flex")
            #     }
            # }
            tensor_programs.append(_tp)

        for idx, (k, d) in enumerate(untemplateable):
            identifier = model_name + "_" + str(idx) + "_" + k
            if d["has_complex_op"]:
                _tp = TensorProgram(
                    identifier,
                    k,
                    d["io_tensors"],
                    d["schedule_gen_func"],
                    d["args"],
                    target,
                    target_host,
                    io_tensors=d["io_tensors"]
                )
                # tp = {
                #     "key": k,
                #     "def": d,
                #     "kind": "untemplateable",
                #     "tasks": {
                #         "sketch": Tasker.task_from_tensor_program(_tp, "sketch"),
                #         "flex": Tasker.task_from_tensor_program(_tp, "flex")
                #     }
                # }
                tensor_programs.append(_tp)
        return tensor_programs

    @staticmethod
    def extract_tasks(module, params, model_name, target, target_host):
        if isinstance(target, str):
            target = Target(target)
        if isinstance(target_host, str):
            target_host = Target(target_host)
        if isinstance(module, Function):
            module = IRModule.from_expr(module)
        assert isinstance(module, IRModule), "only support relay Module or Function to be tuned"

        tpenv = TensorProgramExtractionEnvironment()
        trace_module(module, params, target, tpenv)
        defs = tpenv.tensor_program_definitions

        templateable = []
        untemplateable = []
        for k, d in defs.items():
            if d["schedule_gen_func"] is not None:
                templateable.append((k, d))
            else:
                untemplateable.append((k, d))

        tensor_programs = []

        for idx, (k, d) in enumerate(templateable):
            identifier = model_name + "_" + str(idx) + "_" + k
            _tp = TensorProgram(
                identifier,
                k,
                d["compute_gen_func"],
                d["schedule_gen_func"],
                d["args"],
                target,
                target_host,
                is_topi=True,
                io_tensors=d["io_tensors"]
            )
            tp = {
                "key": k,
                "def": d,
                "kind": "templateable",
                "tasks": {
                    "template": Tasker.task_from_tensor_program(_tp, "template"),
                    "sketch": Tasker.task_from_tensor_program(_tp, "sketch"),
                    "flex": Tasker.task_from_tensor_program(_tp, "flex")
                }
            }
            tensor_programs.append(tp)

        for idx, (k, d) in enumerate(untemplateable):
            identifier = model_name + "_" + str(idx) + "_" + k
            if d["has_complex_op"]:
                _tp = TensorProgram(
                    identifier,
                    k,
                    d["io_tensors"],
                    d["schedule_gen_func"],
                    d["args"],
                    target,
                    target_host,
                    io_tensors=d["io_tensors"]
                )
                tp = {
                    "key": k,
                    "def": d,
                    "kind": "untemplateable",
                    "tasks": {
                        "sketch": Tasker.task_from_tensor_program(_tp, "sketch"),
                        "flex": Tasker.task_from_tensor_program(_tp, "flex")
                    }
                }
                tensor_programs.append(tp)
        return tensor_programs
        

    @staticmethod
    def check_registered_compute(key, _type):
        if _type == "template":
            return key in TEMPLATE_TASK_TABLE
        elif _type == "sketch":
            return key in SKETCH_FUNC_REGISTRY
        elif _type == "flex":
            return key in FLEX_TASK_TABLE
        else:
            return False

    @staticmethod
    def register_compute(key, comp, temp, _type):
        if _type == "sketch":
            if key == None:
                register_workload(comp, override=True)
            else:
                register_workload(key, f=comp, override=True)
        elif _type == "flex":
            flex_register_compute(key, comp, override=True)
        elif _type == "template":
            if key not in TEMPLATE_TASK_TABLE:
                if temp and not comp:
                    create_template(key, temp)
                elif comp and temp:
                    register_topi_compute(key, func=comp)
                    register_topi_schedule(key, func=temp)

    @staticmethod
    def task_from_tensor_program(tensor_program, task_type):
        if isinstance(tensor_program.target, str):
            target = Target(tensor_program.target)
        else:
            target = tensor_program.target

        if task_type == "flex":
            if tensor_program.is_topi:
                comp_def = generic_topi_unwrapper(tensor_program.compute_gen_func, target, tensor_program.func_name)
            else:
                comp_def = tensor_program.compute_gen_func
            compute = flex_wrapper(comp_def)

            tsk = FlexTask(tensor_program.func_name, tensor_program.func_name, compute, tensor_program.args, target, tensor_program.target_host)
            setattr(tsk, 'func_name', tensor_program.func_name)
            setattr(tsk, 'identifier', tensor_program.identifier)
            Tasker.register_compute(tsk.key, tsk, None, "flex")
            return tsk

        elif task_type == "sketch":
            if tensor_program.is_topi:
                comp_def = generic_topi_unwrapper(tensor_program.compute_gen_func, target, tensor_program.func_name)
            else:
                comp_def = tensor_program.compute_gen_func
            compute = sketch_wrapper(comp_def)

            Tasker.register_compute(tensor_program.func_name, compute, None, "sketch")
            with Target(tensor_program.target):
                tsk = SketchTask(
                    func=compute, args=tensor_program.args,
                    target=tensor_program.target, target_host=tensor_program.target_host,
                    hardware_params=None, layout_rewrite_option=None
                )
            setattr(tsk, 'func_name', tensor_program.func_name)
            setattr(tsk, 'identifier', tensor_program.identifier)
            return tsk

        elif task_type == "template":
            Tasker.register_compute(tensor_program.func_name, tensor_program.compute_gen_func, tensor_program.comp_sched_gen_func, "template")
            ser_args = serialize_args(tensor_program.args)
            tsk = TemplateTask(tensor_program.func_name, ser_args)

            tsk.config_space = ConfigSpace()
            with SingleConfigContext(tsk.config_space):
                with target:
                    sch, _ = tsk.func(*ser_args)
                    tsk.config_space.code_hash = getattr(sch, "code_hash", None)
    
            tsk.flop = tsk.config_space.flop or template_compute_flop(sch)
            tsk.target = target
            tsk.target_host = tensor_program.target_host
            setattr(tsk, 'func_name', tensor_program.func_name)
            setattr(tsk, 'identifier', tensor_program.identifier)
            return tsk

        else:
            return None
    
    @staticmethod
    def task_args_repr(task):
        ttype = Tasker.task_type(task)
        if ttype == "template":
            return str(deserialize_args(task.args))
        elif ttype == "sketch":
            tgt = task.target if not isinstance(task.target, str) else Target(task.target)
            with tgt:
                return str(workload_key_to_tensors(task.workload_key))
        elif ttype == "flex":
            return str(deserialize_args(task.args))
        else:
            return None

    @staticmethod
    def task_args(task):
        ttype = Tasker.task_type(task)
        if ttype == "template":
            return deserialize_args(task.args)
        elif ttype == "sketch":
            return decode_workload_key(task.workload_key)[1]
        elif ttype == "flex":
            return task.args
        else:
            return None

    @staticmethod
    def task_repr(task):
        if Tasker.task_type(task) == "template":
            return str(task.workload)
        else:
            return str(task)

    @staticmethod
    def task_type(task):
        if isinstance(task, FlexTask):
            return "flex"
        elif isinstance(task, SketchTask):
            return "sketch"
        elif isinstance(task, TemplateTask):
            return "template"
    
    @staticmethod
    def task_compute_def(task):
        ttype = Tasker.task_type(task)
        if ttype == "template":
            return task.func.fcompute
        elif ttype == "sketch":
            return SKETCH_FUNC_REGISTRY[task.workload_key]
        elif ttype == "flex":
            return FLEX_TASK_TABLE[task.key].func
        else:
            return None

    @staticmethod
    def task_to_key(task):
        ttype = Tasker.task_type(task)
        if ttype == "template":
            return task.name
        elif ttype == "sketch":
            return task.workload_key
        elif ttype == "flex":
            return task.key
        else:
            return None
    
    @staticmethod
    def task_template_def(task):
        if Tasker.task_type(task) == "template":
            return task.func.fschedule
        else:
            return None

    @staticmethod
    def task_theoretical_flop(task):
        try:
            ttype = Tasker.task_type(task)
            if ttype == "template":
                return float(task.flop)
            elif ttype == "sketch":
                return task.compute_dag.flop_ct
            elif ttype == "flex":
                ops, _ = task.func(*task.args)
                s = tvm.te.create_schedule(ops)
                dag = ComputeDAG(s)
                return float(dag.get_flop())
            else:
                return None
        except Exception as e:
            return None

# For completeness
@register_func("auto_scheduler.enter_layout_rewrite", override=True)
def enter_layout_rewrite():
    """Enter layout rewrite tracing environment"""
    env = TensorProgramExtractionEnvironment(TracingMode.PREPARE_LAYOUT_REWRITE)
    env.__enter__()

# For completeness
@register_func("auto_scheduler.exit_layout_rewrite", override=True)
def exit_layout_rewrite():
    """Exit layout rewrite tracing environment"""
    env = TensorProgramExtractionEnvironment.current
    env.__exit__(None, None, None)

@register_func("auto_scheduler.relay_integration.auto_schedule_topi_compute", override=True)
def auto_schedule_topi(func_name, outs):
    io_tensors, has_layout_free, has_complex_op = traverse_to_get_io_tensors(outs)
    if not io_tensors:  # The compute includes dynamic shapes which are not supported yet.
        return None

    try:
        dag = ComputeDAG(io_tensors)
    except tvm.error.TVMError as err:
        print("Failed to create a ComputeDAG for auto_scheduler: %s", str(err))
        return None

    coded_key = register_workload_tensors(dag.workload_key(), io_tensors)
    decoded_key, args = decode_workload_key(coded_key)
    input_map = prepare_input_map(io_tensors)
    target = Target.current()

    tracing_env = TensorProgramExtractionEnvironment.current
    dispatch_ctx = DispatchContext.current
    schedule = None

    # we will have 2 distinct parts to this function
    # 1. Tracing - collecting information about tensor program
    # 2. Scheduling - building schedules from configs and tensor program information

    # 1. Tracing
    if tracing_env is not None:
        if tracing_env.tracing_mode in [TracingMode.EXTRACT_TASK, TracingMode.EXTRACT_COMPLEX_TASK_ONLY]:
            # in the task extraction mode
            # if has_complex_op or tracing_env.tracing_mode == TracingMode.EXTRACT_TASK:
            if has_complex_op:
                tracing_env.add_tensor_program(
                    func_name,
                    coded_key,
                    target,
                    io_tensors,
                    args,
                    input_map,
                    has_layout_free,
                    has_complex_op
                )
                
        elif tracing_env.tracing_mode == TracingMode.PREPARE_LAYOUT_REWRITE:
            # in prepare_layout_rewrite mode
            if (LayoutRewriteOption.get_target_default(target, True) != LayoutRewriteOption.NO_REWRITE and has_layout_free):
                print("rewriting layout")
                # TODO: this needs figuring out
                # state = dispatch_ctx.query(target, key, has_complex_op, dag, func_name)
                state = dispatch_ctx.query(target, coded_key)
                if state is None:
                    return None

                # rewrite the layout and update the context for the new dag
                new_dag = dag.rewrite_layout_from_state(state)
                new_key = new_dag.workload_key()
                if new_key != coded_key:
                    dispatch_ctx.update(target, new_key, state)
        else:
            raise ValueError("Invalid tracing mode: " + tracing_env.tracing_mode)
    
    # 2. Scheduling
    else:
        print("SCHEDULING IN CONDUCTOR")
        new_key = json.dumps([decoded_key] + list(args))
        state = dispatch_ctx.query(target, new_key)

        if isinstance(state, FallbackConfigEntity):
            schedule = None
        else:
            schedule = dispatch_ctx.schedule_schedule(
                func_name, 
                coded_key, 
                decoded_key, 
                target, 
                io_tensors,
                args, 
                dag, 
                input_map, 
                state, 
                has_layout_free, 
                has_complex_op
            )
    return schedule

@register_func("auto_scheduler.relay_integration.te_compiler_update_weights", override=True)
def te_compiler_update_weights(function_weights):
    env = TensorProgramExtractionEnvironment.current
    if env is not None:
        for fname, w in function_weights.items():
            if fname not in env.func_name_to_key:
                continue
            else:
                wkldkey = env.func_name_to_key[fname]
                env.tensor_program_definitions[wkldkey]["weight"] += w
                env.tensor_program_definitions[wkldkey]["func_names"].add(fname)