import os
import time
import tempfile
import cloudpickle
import traceback
import nvtx
from tvm.contrib import tar
from tvm.driver.build_module import build
from tvm.ir.transform import PassContext
from tvm.autotvm.env import AutotvmGlobalScope
from tvm.target import Target
from conductor.utils import gpu_verify
from conductor.mediation import MeasureErrorNo, MeasureInput

def build_routine(_input):
    rng = nvtx.start_range("build_worker_routine", color="yellow")
    tic = time.time()
    orch_scheduler = _input["orch_scheduler"]
    task = _input["task"]
    config = _input["config"]
    options = _input["options"]
    sm = _input["sm"]
    verify = _input["verify"]
    hash_value = "ERRORED"
    hash_callback = _input["hash_callback"]
    try:
        if task == "magic":
            measure_input = MeasureInput.deserialize(cloudpickle.loads(_input["measure_input"]))
            config = measure_input.state
            task = measure_input.task
        elif task == "flex":
            task = cloudpickle.loads(_input["measure_input"])
            config = config["config"]
        else:
            config = config["config"]
        schedule, args, error_no, error_msg = orch_scheduler.from_task(task, config, task.target, options=options)
        if hash_callback != None:
            hash_value = orch_scheduler.encode_schedule(schedule)
            should_skip = hash_callback.hash_callback(config, hash_value)
            if should_skip:
                error_no = MeasureErrorNo.SKIP
                error_msg = "Skipped due to hash"

        if error_no != MeasureErrorNo.NO_ERROR:
            filename = ""
            args = None
        else:
            filename = os.path.join(tempfile.mkdtemp(), str(options["idx"]) + "." + tar.tar.output_format)
            tgt = task.target if not isinstance(task.target, str) else Target(task.target)
            if "cuda" in tgt.keys:
                AutotvmGlobalScope.current.cuda_target_arch = sm
                pass_ctx_conf = {"tir.add_lower_pass": [(2, gpu_verify(**verify["check_gpu"]))]}
            else:
                pass_ctx_conf = None
            with tgt:
                with PassContext(config=pass_ctx_conf) if pass_ctx_conf else PassContext():
                    func = build(schedule, args, target=tgt, target_host=task.target_host)
            func.export_library(filename, tar.tar)
    except Exception as e:
        error_no = MeasureErrorNo.INSTANTIATION_ERROR
        error_msg = traceback.format_exc()

    tac = time.time()
    nvtx.end_range(rng)
    return (filename, args, error_no, error_msg, tac - tic, tac)