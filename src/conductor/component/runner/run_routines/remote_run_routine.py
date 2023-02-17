from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
import time
import shutil
import os
from tvm.runtime.ndarray import empty
from conductor.utils import make_traceback_info, get_const_tuple, request_remote
from conductor.mediation import MeasureErrorNo

def run_routine(_input):
    _build_res = _input["_build_res"]
    flop = _input["flop"]
    target = _input["target"]
    device_id = _input["device_id"]
    cooldown_interval = _input["cooldown_interval"]
    evaluator = _input["evaluator"]
    dev_ctx_details = _input["dev_ctx_details"]
    remove_built_schedule = _input["remove"]

    tic = time.time()
    error_no = MeasureErrorNo.NO_ERROR
    error_msg = None
    remote = None
    other_out = None

    try:
        newctx = dev_ctx_details
        remote = request_remote(*newctx)
        dev = remote.device(target, device_id)
        
        # added in-line with mainline tvm
        # supposedly helps to recover from errors since we're running in non-0 stream this way
        # streams will still be serialized at the GPU unless running with NVIDIA MPS
        stream = dev.create_raw_stream()
        dev.set_raw_stream(stream)

        if target == "cuda":
            cuda_arch = "sm_" + "".join(dev.compute_version.split("."))
            set_cuda_target_arch(cuda_arch)
        try:
            random_fill = remote.get_function("tvm.contrib.random.random_fill")
        except AttributeError:
            raise AttributeError("Please make sure USE_RANDOM is ON in the config.cmake " "on the remote devices")
        args = [empty(get_const_tuple(
            x.shape), x.dtype, dev) for x in _build_res.args]
        for arg in args:
            random_fill(arg)
        remote.upload(_build_res.filename)
        func = remote.load_module(os.path.split(_build_res.filename)[1])
        dev.sync()
        func.entry_func(*args)
        dev.sync()
        costs, mean, total_time, other = evaluator.evaluate(
            func,
            func.entry_name,
            flop,
            dev,
            args,
            fname=_build_res.filename.split("/")[-1].split(".")[0]
        )
        achieved_flop = flop / mean
        dev.free_raw_stream(stream)
        other_out = other
    except Exception:
        dev.free_raw_stream(stream)
        mean = 1e20
        total_time = 1e20
        costs = (1e20,)
        achieved_flop = 0
        error_no = MeasureErrorNo.RUNTIME_DEVICE
        error_msg = make_traceback_info()
        print(error_msg)

    toc = time.time()
    time.sleep(cooldown_interval)
    if remove_built_schedule:
        if os.path.exists(os.path.dirname(_build_res.filename)):
            shutil.rmtree(os.path.dirname(_build_res.filename))
        if remote is not None:
            remote.remove(_build_res.filename)
            remote.remove(os.path.splitext(_build_res.filename)[0] + ".so")
            remote.remove("")

    return (costs, error_no, error_msg, toc - tic + _build_res.time_cost, toc, mean, total_time, other_out, achieved_flop)