from conductor._base import Configurable
from conductor.component.evaluator._base import Evaluator

import logging
logger = logging.getLogger("conductor.component.evaluator.default")

class DefaultEvaluator(Evaluator):
    _name = "default"
    
    def __repr__(self):
        return Evaluator.__repr__(self) + ":" + DefaultEvaluator._name

    def __init__(self, configs=None):
        Evaluator.__init__(self, ["default"], configs=configs, child_default_configs=Configurable.merge_configs({
            "num_avg_runs": 10,
            "num_measure_repeat": 1,
            "min_repeat_ms": 0,
            "enable_cpu_cache_flush": False
        }, {}, override_first=True))

    def get_ctx(self):
        return super().get_ctx()

    def evaluate(self, mod_func, mod_func_name, flop, ctx, args, fname=None):
        time_f = self.get_func(mod_func, mod_func_name, ctx)
        if args is not None:
            e = time_f(*args)
        else:
            e = time_f()
        return (e.results, e.mean, e.total_time, e.other)