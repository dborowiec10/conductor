import tvm
from conductor.component.method.sketch import SketchMethod
from conductor.utils import array_mean, get_const_tuple
from conductor.mediation import MeasureErrorNo
from conductor.experiment import Experiment

import logging
logger = logging.getLogger("conductor.component.method.composite_sketch")

# Tuning method which relies on policy - Ansor
class CompositeSketchMethod(SketchMethod):
    _name = "composite"
    
    def __repr__(self):
        return SketchMethod.__repr__(self) + ":" + CompositeSketchMethod._name

    def __init__(self, orch_scheduler, cost_model, search_policy, configs=None, profilers_specs=[]):
        SketchMethod.__init__(
            self, 
            ["composite"], 
            orch_scheduler,
            configs=configs, 
            child_default_configs={}, 
            profilers_specs=profilers_specs
        )

        self.cost_model = cost_model
        self.search_policy_cl = search_policy
        self.search_policy = None
        self.configs = configs

    def load(self, task, configurer):
        SketchMethod.load(self, task, configurer)
        if self.search_policy_cl:
            self.search_policy = self.search_policy_cl(self.task, self.cost_model, configs=self.configs)

    def unload(self):
        SketchMethod.unload(self)

    def execute(self, num_measurements):
        Experiment.current.set_experiment_stage("method:execute")
        self.profiling_checkpoint("execute:start")

        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = True
        tvm.autotvm.env.GLOBAL_SCOPE.silent = True

        measure_inputs, measure_results = self.search_policy.continue_search_one_round(num_measurements, self.measurer.program_measurer)
        Experiment.current.set_experiment_stage("method:execute")
        if len(measure_inputs) < 1:
            self.profiling_checkpoint("execute:stop")

            return {
                "flop": None, "config": None,
                "pair": None, "idx": None
            }, {
                "cost": None, "config": None,
                "pair": None, "idx": None
            }, -1, [], []

        best_flop = 0
        best_flop_config = None
        best_flop_idx = None
        best_flop_pair = None

        best_cost = 1e10
        best_cost_config = None
        best_cost_idx = None
        best_cost_pair = None

        err_cnt = 0
        for idx, (measure_input, measure_result) in enumerate(zip(measure_inputs, measure_results)):
            flops = measure_result.achieved_flop
            if measure_result.error_no == MeasureErrorNo.NO_ERROR:
                cost = float(array_mean(get_const_tuple(measure_result.costs)))
            else:
                cost = 1e20
            if cost < best_cost:
                best_cost = cost
                best_cost_pair = (measure_input, measure_result)
                best_cost_idx = idx
            if flops > best_flop:
                best_flop = flops
                best_flop_pair = (measure_input, measure_result)
                best_flop_idx = idx
            if measure_result.error_no != 0:
                err_cnt += 1

        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = False
        tvm.autotvm.env.GLOBAL_SCOPE.silent = False

        self.profiling_checkpoint("execute:stop")

        return ({
            "flop": best_flop, "config": best_flop_config,
            "pair": best_flop_pair, "idx": best_flop_idx
        }, {
            "cost": best_cost, "config": best_cost_config,
            "pair": best_cost_pair, "idx": best_cost_idx
        }, err_cnt, measure_inputs, measure_results)