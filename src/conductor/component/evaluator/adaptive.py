import numpy as np
from conductor._base import Configurable
from conductor.component.evaluator._base import Evaluator

import logging
logger = logging.getLogger("conductor.component.evaluator.adaptive")

class AdaptiveEvaluator(Evaluator):
    _name = "adaptive"
    
    def __repr__(self):
        return Evaluator.__repr__(self) + ":" + AdaptiveEvaluator._name

    def __init__(self, configs=None):
        Evaluator.__init__(self, ["adaptive"], configs=configs, child_default_configs=Configurable.merge_configs({
            "num_avg_runs": 10,
            "num_measure_repeat": 1,
            "min_repeat_ms": 0,
            "enable_cpu_cache_flush": False,
            "epsilon": 0.1,
            "batch_size": 50,
            "min_repeat_till_active": 300
        }, {}, override_first=True))
        self.epsilon = self.config["epsilon"]
        self.batch_size = self.config["batch_size"]
        self.minimum_repeats_to_activate = self.config["min_repeat_till_active"]

    def get_ctx(self):
        supctx = super().get_ctx().copy()
        supctx.update({
            "epsilon": self.epsilon,
            "batch_size": self.batch_size,
            "min_repeat_till_active": self.minimum_repeats_to_activate
        })
        return supctx

    def evaluate(self, mod_func, mod_func_name, flop, ctx, args, fname=None):
        # as per original implementation, won't turn it on unless we have at least 300 repeats
        if self.repeat * self.number < self.minimum_repeats_to_activate:
            time_f = self.get_func(mod_func, mod_func_name, ctx)
            e = time_f(*args)
            return (e.results, e.mean, e.total_time, e.other)
        else:
            _number = self.number
            _repeat = self.repeat
            b_size = self.batch_size
            costs = []
            sum_num = 0
            max_iter = _number * _repeat
            pis = []
            bi = 1
            flag = True
            total_time = 0
            others = []
            while flag and sum_num < max_iter:
                self.number = b_size
                self.repeat = 1
                time_f = self.get_func(mod_func, mod_func_name, ctx)
                eval_res = time_f(*args)
                others.append(eval_res.other)
                b_mean = eval_res.mean
                total_time = total_time + eval_res.total_time
                costs.append(b_mean)
                sum_num = sum_num + b_size
                pi = flop / b_mean
                pis.append(pi)
                pis_array = np.array(pis)
                if len(pis_array) > 4:
                    pis_array.sort()
                    pis_array = pis_array[1:-1]
                cv = pis_array.std()/pis_array.mean()
                if bi > 2 and cv < self.epsilon:
                    flag = False
                bi = bi + 1

            self.number = _number
            self.repeat = _repeat
            return (costs, sum(costs) / float(len(costs)), total_time, others)