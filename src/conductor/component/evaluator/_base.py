from conductor._base import Configurable
from conductor.mediation import ProfileResult

import tvm
import json
import numpy as np

import logging
logger = logging.getLogger("conductor.component.evaluator")

class Evaluator(Configurable):
    _name = "evaluator"
    
    def __repr__(self):
        return Evaluator._name

    def __init__(self, subtypes, configs=None, child_default_configs={}):
        Configurable.__init__(self, "evaluator", subtypes, configs, Configurable.merge_configs({
            "num_avg_runs": 10,
            "num_measure_repeat": 1,
            "min_repeat_ms": 0,
            "enable_cpu_cache_flush": False
        }, child_default_configs, override_first=True))
        self.f_preproc = 'cache_flush_cpu_non_first_arg' if self.config.get("enable_cpu_cache_flush", False) else ''
        self.number = self.config["num_avg_runs"]
        self.repeat = self.config["num_measure_repeat"]
        self.min_repeat_ms = self.config["min_repeat_ms"]

    def get_ctx(self):
        return {
            "num_avg_runs": self.number,
            "num_measure_repeat": self.repeat,
            "min_repeat_ms": self.min_repeat_ms,
            "f_preproc": self.f_preproc,
            "enable_cpu_cache_flush": self.config.get("enable_cpu_cache_flush", False)
        }

    def evaluate(self, mod_func, mod_func_name, flop, ctx, args, fname=None):
        raise NotImplementedError()

    def get_func(self, mod_func, mod_func_name, ctx):
        try:
            fcreate = tvm._ffi.get_global_func("runtime.RPCTimeEvaluator")
            feval = fcreate(
                mod_func,
                mod_func_name, 
                ctx.device_type,
                ctx.device_id, 
                self.number, 
                self.repeat, 
                self.min_repeat_ms, 
                self.f_preproc
            )
            def evaluator(*args):
                """Internal wrapped evaluator."""
                blob = feval(*args)
                out = json.loads(blob)
                return ProfileResult(mean=np.mean(out["costs"]), results=out["costs"], total_time=out["total_time"], other=out["other"])     
            return evaluator
        except NameError:
            raise NameError("time_evaluate is only supported when RPC is enabled")
