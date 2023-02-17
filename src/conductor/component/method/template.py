
from conductor._base import Configurable
from conductor.component.method._base import Method

import tvm

import logging

from conductor.experiment import Experiment
logger = logging.getLogger("conductor.component.method.template")

class TemplateMethod(Method):
    _name = "template"
    
    def __repr__(self):
        return Method.__repr__(self) + ":" + TemplateMethod._name

    def __init__(self, subtypes, orch_scheduler, configs=None, child_default_configs={}, profilers_specs=[]):
        Method.__init__(
            self, 
            ["template"] + subtypes,
            "template",
            orch_scheduler, 
            configs=configs,
            child_default_configs=Configurable.merge_configs({}, child_default_configs, override_first=True),
            profilers_specs=profilers_specs
        )
        self.best_flops = 0
    
    def load(self, task, configurer):
        Method.load(self, task, configurer)

    def unload(self):
        Method.unload(self)

    def update(self, inputs, results):
        raise NotImplementedError()

    def has_next(self):
        raise NotImplementedError()

    def next_batch(self, count):
        raise NotImplementedError()

    def execute(self, num_measurements):
        Experiment.current.set_experiment_stage("method:execute")
        self.profiling_checkpoint("execute:start")

        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = True
        tvm.autotvm.env.GLOBAL_SCOPE.silent = True

        if not self.has_next():
            self.profiling_checkpoint("execute:stop")
            
            return ({
                "flop": None, "config": None,
                "pair": None, "idx": None
            }, {
                "cost": None, "config": None,
                "pair": None, "idx": None
            }, -1, [], [])

        # fetch a batch of configurations from the method
        configurations = self.next_batch(num_measurements)
        # measure configurations
        measure_inputs, measure_results, err_cnt = self.measurer.measure(self.task, configurations)
        Experiment.current.set_experiment_stage("method:execute")
        best_flop = 0
        best_flop_config = None
        best_flop_idx = None
        best_flop_pair = None
        best_cost = 1e10
        best_cost_config = None
        best_cost_idx = None
        best_cost_pair = None

        # for legacy reasons, we keep track of self.best_flops internally
        # should really be managed from within the strategy
        # however, some filters use it for filtration
        # choose best out of the batch
        for idx, (measure_input, measure_result) in enumerate(zip(measure_inputs, measure_results)):
            flops = measure_result.achieved_flop
            mean_cost = measure_result.mean
            if flops > best_flop:
                best_flop = flops
                self.best_flops = best_flop
                best_flop_config = measure_input.config
                best_flop_pair = (measure_input, measure_result)
                best_flop_idx = idx                            
            if mean_cost < best_cost:
                best_cost = mean_cost
                best_cost_config = measure_input.config
                best_cost_pair = (measure_input, measure_result)
                best_cost_idx = idx

        # update method
        self.update(measure_inputs, measure_results)

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