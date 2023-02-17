import requests
import os
import tqdm
import numpy as np
import hashlib
import json
import string
import traceback
import random
from typing import Dict, Any
from tvm.autotvm.task.space import InstantiationError, ConfigSpace
from tvm.tir.analysis import verify_gpu_code
from tvm.tir.transform import prim_func_pass, Simplify
from tvm.tir.expr import IntImm, FloatImm, Any
from tvm.tir import Var
from tvm.rpc import connect_tracker
from tvm.ir.transform import Sequential
from tvm.target import Target
from tvm import gpu, cpu

def check_definition(_dict, definition):
    if ":" in definition:
        defsplits = definition.split(":")
        locdict = _dict
        missing = False
        for d in defsplits:
            if d in locdict:
                locdict = locdict[d]
            else:
                missing = True
                break
        return not missing
    else:
        if definition in _dict:
            return True
        else:
            return False

def match_definition(_dict, definition):
    if ":" in definition:
        defsplits = definition.split(":")
        locdict = _dict
        missing = False
        for d in defsplits:
            if d in locdict:
                locdict = locdict[d]
            else:
                missing = True
                break
        if missing:
            return None
        else:
            return locdict
    else:
        if definition in _dict:
            return _dict[definition]
        else:
            return None

def download(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes or not os.path.isfile(save_path):
        raise Exception(f"Something went wrong while downloading file from {url}")

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def hash(_obj):
    """MD5 hash of a _obj."""
    dhash = hashlib.md5()
    encoded = json.dumps(_obj, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def random_string(len):
     return ''.join([random.choice(string.ascii_lowercase + string.digits) for i in range(len)])

def array_mean(arr):
    return sum(x for x in arr) / len(arr)

def point2knob(p, dims):
    """convert point form (single integer) to knob form (vector)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    return knob

def knob2point(knob, dims):
    """convert knob form (vector) to point form (single integer)"""
    p = 0
    for j, k in enumerate(knob):
        p += int(np.prod(dims[:j])) * k
    return p

def gpu_verify(**kwargs):
    def verify_pass(f, *_):
        valid = verify_gpu_code(f, kwargs)
        if not valid:
            raise InstantiationError("Skipped because of invalid gpu kernel")
        return f
    return prim_func_pass(verify_pass, opt_level=0)

def sample_ints(low, high, m):
    vis = set()
    assert m <= high - low
    while len(vis) < m:
        new = random.randrange(low, high)
        while new in vis:
            new = random.randrange(low, high)
        vis.add(new)

    return list(vis)

def request_remote(device_key, host=None, port=None, priority=1, timeout=60):
    # connect to the tracker
    host = host or os.environ["TVM_TRACKER_HOST"]
    port = port or int(os.environ["TVM_TRACKER_PORT"])
    tracker = connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout)
    return remote

# unwraps topi-defined functions by re-wrapping them
# the wrapper will return the output tensors of the function
# internally, it will call the compute_def function with empty 
# config to satisfy parameters and pass through any function params
# WARNING: this should only be called on a topi-defined function
# it has no way of checking whether the function is actually defined in topi
# so... be careful!
def generic_topi_unwrapper(func, tgt, name):
    def wrapper(*args, **kwargs):
        if isinstance(tgt, Target):
            t = tgt
        else:
            t = Target(tgt)
        with t:
            out = func(ConfigSpace(), *args, **kwargs)
            return out
    setattr(wrapper, 'func_name', name)
    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper

def generic_topi_schedule_unwrapper(func, tgt, name):
    def wrapper(*args, **kwargs):
        if isinstance(tgt, Target):
            t = tgt
        else:
            t = Target(tgt)
        with t:
            sched = func(ConfigSpace(), *args)
            return sched
    setattr(wrapper, 'func_name', name)
    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper

def get_sm(self, dev_ctx_details, target, device_id):
    if isinstance(target, str):
        tgt = Target(target)
    else:
        tgt = target

    if dev_ctx_details is not None:
        remote = request_remote(*dev_ctx_details)
        ctx = remote.context("cuda" if "cuda" in tgt.keys else "llvm", device_id)
    else:
        if "cuda" in tgt.keys:
            ctx = gpu(device_id)
        else:
            ctx = cpu(device_id)
    if ctx.exist:
        if ("cuda" in tgt.keys or "opencl" in tgt.keys or "rocm" in tgt.keys or "vulkan" in tgt.keys):
            sm = ("sm_" + "".join(ctx.compute_version.split(".")))
        else:
            sm = None
    else:
        raise RuntimeError("Could not find a context!")
    
    return sm


def make_traceback_info(length=512):
    info = str(traceback.format_exc())
    if len(info) > length:
        info = (
            info[: length // 2] + "\n...\n" + info[-length // 2 :]
        )
    return info



def get_const_int(exp):
    """Verifies expr is integer and get the constant value.
    Parameters
    ----------
    exp : Union[tvm.tir.expr, int]
        The input expression.
    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(exp, int):
        return exp
    if not isinstance(exp, IntImm):
        opt = Sequential([Simplify()])
        exp = opt(exp)
    if not isinstance(exp, IntImm):
        raise ValueError("Expect value to be constant int")
    return exp.value

def get_const_float(exp):
    """Verifies expr is float and get the constant value.

    Parameters
    ----------
    exp : Union[tvm.tir.expr, float]
        The input expression.

    Returns
    -------
    out_value : float
        The output.
    """
    if isinstance(exp, int):
        return exp
    if not isinstance(exp, FloatImm):
        opt = Sequential([Simplify()])
        exp = opt(exp)
    if not isinstance(exp, FloatImm):
        raise ValueError("Expect value to be constant int")
    return exp.value

def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int.

    Parameters
    ----------
    in_tuple : Tuple[tvm.tir.expr]
        The input.

    Returns
    -------
    out_tuple : Tuple[Union[int,tvm.tir.Var,tvm.tir.Any]]
        The output tuple of int. The dynamic shape variables (Var or Any) will be preserved.
    """
    ret = []
    for elem in in_tuple:
        if isinstance(elem, (Var, Any)):
            ret.append(elem)
        else:
            if isinstance(elem, FloatImm):
                ret.append(get_const_float(elem))
            elif isinstance(elem, IntImm):
                ret.append(get_const_int(elem))
    return tuple(ret)


def get_input_info(graph, params_dict):
    shape_dict = {}
    dtype_dict = {}
    param_names = [k for (k, v) in params_dict.items()]
    for node_id in graph["arg_nodes"]:
        node = graph["nodes"][node_id]
        name = node["name"]
        if name not in param_names:
            shape_dict[name] = graph["attrs"]["shape"][1][node_id]
            dtype_dict[name] = graph["attrs"]["dltype"][1][node_id]
    return shape_dict, dtype_dict

def generate_tensor_data(shape, dtype, fill_mode):
    if fill_mode == "zeros":
        tensor = np.zeros(shape=shape, dtype=dtype)
    elif fill_mode == "ones":
        tensor = np.ones(shape=shape, dtype=dtype)
    elif fill_mode == "random":
        if "int8" in dtype:
            tensor = np.random.randint(128, size=shape, dtype=dtype)
        else:
            tensor = np.random.uniform(-1, 1, size=shape).astype(dtype)
    else:
        raise RuntimeError("LOLOLOLO")
    return tensor

def make_inputs_dict(shape_dict, dtype_dict, inputs=None, fill_mode="random"):
    if inputs is None:
        inputs = {}
    for input_name in inputs:
        if input_name not in shape_dict.keys():
            raise RuntimeError("LOLOLOLO")
    inputs_dict = {}
    for input_name in shape_dict:
        if input_name in inputs.keys():
            inputs_dict[input_name] = inputs[input_name]
        else:
            shape = shape_dict[input_name]
            dtype = dtype_dict[input_name]
            data = generate_tensor_data(shape, dtype, fill_mode)
            inputs_dict[input_name] = data
    return inputs_dict