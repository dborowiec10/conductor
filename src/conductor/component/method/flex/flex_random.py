from conductor._base import Configurable
from conductor.component.method.flex.flex import FlexMethod
import math
import logging
logger = logging.getLogger("conductor.component.method.rndflex")


class FlexRandomMethod(FlexMethod):
    _name = "rndflex"
    
    def __repr__(self):
        return FlexMethod.__repr__(self) + ":" + FlexRandomMethod._name

    def __init__(self, orch_scheduler, configs=None, profilers_specs=[]):
        FlexMethod.__init__(
            self,
            ["standalone", "rndflex"],
            orch_scheduler,
            configs=configs,
            child_default_configs=Configurable.merge_configs({
                "force_inline": True,
                "rewrite": False,
                "slevel": 4,
                "rlevel": 3,
                "op_space_groups": 3,
                "op_unroll_policy": "off",
                "op_fuse_policy": "off",
                "op_reorder_policy": "off",
                "op_perf_model_path_lst": [],
                "graph_perf_model_path": None,
                "op_graph_trial_split": [0.8, 0.2],
                "op_graph_early_stop": [15, 5],
                "op_trials": None,
                "graph_trials": None
            }, {}, override_first=True),
            profilers_specs=profilers_specs
        )

    def procedure(self, scheduler, configs, type_keys, mode, config_mode, trials, early_stop, use_model=False):
        if use_model:
            scheduler.walker_group.load_or_create_model()

        result_dictionaries = []
        if mode == "flex":
            for _ in range(trials):
                warm_up_epochs = 1
                warm_up_trials = self.measurer.n_parallel

                res_dicts = self.warm_up(scheduler, warm_up_epochs, warm_up_trials, configs, type_keys, config_mode, use_model=use_model)
                result_dictionaries += res_dicts
            return scheduler.walker_group.to_config(scheduler.walker_group.top1()), result_dictionaries

        elif mode == "manual":
            determined_trials = math.ceil(trials / self.measurer.n_parallel)
            rem = trials % self.measurer.n_parallel
            for _ in range(determined_trials):
                warm_up_epochs = 1
                res_dicts = self.warm_up(scheduler, warm_up_epochs, self.measurer.n_parallel, configs, type_keys, config_mode, use_model=use_model)
                result_dictionaries += res_dicts
            result_dictionaries += self.warm_up(scheduler, 1, rem, configs, type_keys, config_mode, use_model=use_model)
            return scheduler.walker_group.to_config(scheduler.walker_group.top1()), result_dictionaries