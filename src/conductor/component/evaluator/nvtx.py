from conductor._base import Configurable
from conductor.component.evaluator._base import Evaluator
from conductor.mediation import ProfileResult

import tvm
import json
import numpy as np
import logging
logger = logging.getLogger("conductor.component.evaluator.nvtx")

class NVTXEvaluator(Evaluator):
    _name = "nvtx"
    
    def __repr__(self):
        return Evaluator.__repr__(self) + ":" + NVTXEvaluator._name

    def __init__(self, configs=None):
        Evaluator.__init__(self, ["nvtx"], configs=configs, child_default_configs=Configurable.merge_configs({
            "num_avg_runs": 10,
            "num_measure_repeat": 1,
            "min_repeat_ms": 0,
            "enable_cpu_cache_flush": False,
            "eval_kind": "default"
        }, {}, override_first=True))

    def get_ctx(self):
        return super().get_ctx()

    def evaluate(self, mod_func, mod_func_name, flop, ctx, args, fname=None):
        if self.config["eval_kind"] == "default":
            fcreate = tvm._ffi.get_global_func("runtime.RPCTimeEvaluatorNVTX")
        elif self.config["eval_kind"] == "activity":
            fcreate = tvm._ffi.get_global_func("runtime.RPCActivityEvaluatorNVTX")
        feval = fcreate(
            mod_func, mod_func_name, 
            ctx.device_type, ctx.device_id, 
            self.number,
            self.repeat,
            self.min_repeat_ms,
            self.f_preproc,
            fname
        )
        def evaluator(*args):
            """Internal wrapped evaluator."""
            blob = feval(*args)
            out = json.loads(blob)
            return ProfileResult(mean=np.mean(out["costs"]), results=out["costs"], total_time=out["total_time"], other=out["other"])
            
        e = evaluator(*args)
        return (e.results, e.mean, e.total_time, e.other)