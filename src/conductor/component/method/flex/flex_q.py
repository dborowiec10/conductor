from conductor._base import Configurable
from conductor.component.method.flex.flex import FlexMethod

import math
import numpy as np

import logging
logger = logging.getLogger("conductor.component.method.qflex")


class FlexQMethod(FlexMethod):
    _name = "qflex"
    
    def __repr__(self):
        return FlexMethod.__repr__(self) + ":" + FlexQMethod._name

    def __init__(self, orch_scheduler, configs=None, profilers_specs=[]):
        FlexMethod.__init__(
            self,
            ["standalone", "qflex"],
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
        scheduler.walker_group.load_walker_model()
        if use_model:
            scheduler.walker_group.load_or_create_model()

        result_dictionaries = []

        if mode == "flex":
            # warm up
            warm_up_epochs = 10
            warm_up_trials = 20

            # perform warm-up
            logger.info(f"warm_up: epochs: {warm_up_epochs}, trials: {warm_up_trials}")
            res_dicts = self.warm_up(scheduler, warm_up_epochs, warm_up_trials, configs, type_keys, config_mode, use_model=use_model)
            result_dictionaries += res_dicts

            # find 1 best config
            best = scheduler.walker_group.top1()
            best_value = scheduler.walker_group.top1_value()

            retired_indices = []
            value_early_stop = best_value
            early_stop_count = 0
            # find n_parallel best configs
            cur_lst = scheduler.walker_group.topk(self.measurer.n_parallel, modify=True, with_value=True)

            # split trials in 5 parts
            part = math.ceil(trials / 5)

            for trial in range(trials):
                logger.info(f"trial: {trial}")
                from_lst, next_points, action_lst = scheduler.walker_group.walk(cur_lst, trial)
                if use_model:
                    results = scheduler.walker_group.query_performance(next_points)
                else:
                    next_configs = [scheduler.walker_group.to_config(indices) for indices in next_points]
                    logger.info(f"measuring configs: next_configs: len: {len(next_configs)}")
                    results, res_dict = self.measure_configs(scheduler, configs, next_configs, config_mode)
                    result_dictionaries.append(res_dict)
                    scheduler.walker_group.add_perf_data(next_points, results)
                
                for indices, action, (from_indices, from_value), result in zip(next_points, action_lst, from_lst, results):
                    reward = np.tanh(max(from_value - result, 0.0))
                    scheduler.walker_group.add_data(
                        action[0],      # name
                        from_indices,   # pre_state
                        action[1],      # action
                        indices,        # post_state
                        reward          # reward
                    )
                    scheduler.walker_group.record(indices, result, random_reject=True)

                if scheduler.walker_group.top1_value() < best_value:
                    best_value = scheduler.walker_group.top1_value()
                    best = scheduler.walker_group.top1()

                if math.fabs(best_value - value_early_stop) < 0.02:
                    early_stop_count += 1
                else:
                    value_early_stop = best_value
                    early_stop_count = 0

                if early_stop_count >= early_stop:
                    break

                if not scheduler.walker_group.has_more():
                    break

                retired_indices.extend(cur_lst)
                cur_lst = scheduler.walker_group.topk(self.measurer.n_parallel, modify=True, with_value=True)

                # if we have done at least 1/5th of all trials
                if (trial + 1) % part == 0:
                    logger.info("done 1/5th of trials")
                    scheduler.walker_group.train_walkers()

                    if not use_model:
                        if best_value < float("inf"):
                            scheduler.walker_group.record(best, best_value, random_reject=False)
                            best = {}
                            best_value = float("inf")

                        for indices, value in retired_indices[-self.measurer.n_parallel:-1]:
                            scheduler.walker_group.record(indices, value, random_reject=False)

                        indices_lst = scheduler.walker_group.topk(self.measurer.n_parallel, modify=True)
                        next_configs = [scheduler.walker_group.to_config(indices) for indices in indices_lst]
                        logger.info(f"measuring configs: next_configs: len: {len(next_configs)}")
                        results, res_dict = self.measure_configs(scheduler, configs, next_configs, config_mode)
                        result_dictionaries.append(res_dict)
                        scheduler.walker_group.add_perf_data(indices_lst, results)

                        for indices, result in zip(indices_lst, results):
                            scheduler.walker_group.record(indices, result, random_reject=False)

                    warm_up_epochs = 1
                    warm_up_trials = self.measurer.n_parallel
                    logger.info(f"warming up some more: epochs: {warm_up_epochs}, trials: {warm_up_trials}")
                    res_dicts = self.warm_up(scheduler, warm_up_epochs, warm_up_trials, configs, type_keys, config_mode, use_model=use_model)
                    result_dictionaries += res_dicts

                    if scheduler.walker_group.top1_value() < best_value:
                        best_value = scheduler.walker_group.top1_value()
                        best = scheduler.walker_group.top1()

            scheduler.walker_group.clear_data()
            return scheduler.walker_group.to_config(best), result_dictionaries

        elif mode == "manual":
            pass