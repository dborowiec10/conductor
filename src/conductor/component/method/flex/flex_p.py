from conductor._base import Configurable
from conductor.component.method.flex.flex import FlexMethod

import math
import numpy as np

import logging
logger = logging.getLogger("conductor.component.method.pflex")


class FlexPMethod(FlexMethod):
    _name = "pflex"
    
    def __repr__(self):
        return FlexMethod.__repr__(self) + ":" + FlexPMethod._name

    def __init__(self, orch_scheduler, configs=None, profilers_specs=[]):
        FlexMethod.__init__(
            self,
            ["standalone", "pflex"],
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
            warm_up_epochs = 20  # as hardcoded in flex tensor
            warm_up_trials = 20  # as hardcoded in flex tensor
            logger.info(f"warm_up: epochs: {warm_up_epochs}, trials: {warm_up_trials}")
            res_dicts = self.warm_up(scheduler, warm_up_epochs, warm_up_trials, configs, type_keys, config_mode, use_model=use_model)
            result_dictionaries += res_dicts
            minimal = [{}, float("inf")]    # the minimal point found before
            retired_indices = []            # list of local minimals
            part = math.ceil(trials / 20)
            value_early_stop = scheduler.walker_group.top1_value()
            early_stop_count = 0
            count_incessant_empty_trial = 0
            for trial in range(trials):
                logger.info(f"trial: {trial}")
                # in case we don't have enough to perform further, warm up some more
                if not scheduler.walker_group.has_more():
                    warm_up_epochs = 1
                    warm_up_trials = self.measurer.n_parallel
                    logger.info(f"warming up some more: epochs: {warm_up_epochs}, trials: {warm_up_trials}")
                    res_dicts = self.warm_up(scheduler, warm_up_epochs, warm_up_trials, configs, type_keys, config_mode, use_model=use_model)
                    result_dictionaries += res_dicts
                    continue

                from_indices, from_value = scheduler.walker_group.top_random(with_value=True)
                next_indices_lst, action_lst = scheduler.walker_group.full_walk(from_indices, no_repeat=True)
                next_configs = [scheduler.walker_group.to_config(indices) for indices in next_indices_lst]
                if len(next_configs) < 1:
                    count_incessant_empty_trial += 1
                else:
                    count_incessant_empty_trial = 0
                if use_model:
                    results = scheduler.walker_group.query_performance(next_indices_lst)
                else:
                    logger.info(f"measuring configs: next_configs: len: {len(next_configs)}")
                    results, res_dict = self.measure_configs(scheduler, configs, next_configs, config_mode)
                    result_dictionaries.append(res_dict)
                    scheduler.walker_group.add_perf_data(next_indices_lst, results)
                rewards = [np.tanh(max(from_value - result, 0.0)) for result in results]
                is_local_minimal = True
                for indices, action, reward, result in zip(next_indices_lst, action_lst, rewards, results):
                    scheduler.walker_group.add_data(
                        action[0],      # name
                        from_indices,   # pre_state
                        action[1],      # action
                        indices,        # post_state
                        reward          # reward
                    )
                    scheduler.walker_group.record(indices, result, random_reject=True)
                    if result < scheduler.walker_group.top1_value():
                        is_local_minimal = False
                if is_local_minimal or count_incessant_empty_trial > 0:
                    top = scheduler.walker_group.pop_top()
                    if top.value < minimal[1]:
                        if minimal[1] < float("inf"):
                            retired_indices.append(minimal)
                        minimal[1] = top.value
                        minimal[0] = top.indices
                    else:
                        retired_indices.append([top.indices, top.value])
                if scheduler.walker_group.top1_value() < minimal[1]:
                    cur_best_value = scheduler.walker_group.top1_value()
                    cur_best = scheduler.walker_group.top1()
                else:
                    cur_best_value = minimal[1]
                    cur_best = minimal[0]
                if count_incessant_empty_trial >= early_stop:
                    break
                if math.fabs(cur_best_value - value_early_stop) < 0.02:
                    early_stop_count += 1
                else:
                    value_early_stop = cur_best_value
                    early_stop_count = 0
                if early_stop_count >= early_stop:
                    break

                if (trial + 1) % part == 0:
                    logger.info("done 1/5th of trials")
                    if not use_model:
                        if minimal[1] < float("inf"):
                            scheduler.walker_group.record(minimal[0], minimal[1], random_reject=False)
                        for retired in retired_indices:
                            scheduler.walker_group.record(retired[0], retired[1], random_reject=False)

                        minimal[0] = {}
                        minimal[1] = float("inf")
                        reevaluate_number = 10  # as hardcoded in flex tensor
                        indices_lst = scheduler.walker_group.topk(reevaluate_number, modify=True)
                        next_configs = [scheduler.walker_group.to_config(indices) for indices in indices_lst]
                        old_parallel = self.measurer.n_parallel

                        if "cuda" in self.task.target.keys:
                            self.measurer.n_parallel = 1
                        else:
                            # min(self.parallel, os.cpu_count())
                            self.measurer.n_parallel = 1
                        logger.info(f"measuring configs: next_configs: len: {len(next_configs)}")
                        results, res_dict = self.measure_configs(scheduler, configs, next_configs, config_mode)
                        result_dictionaries.append(res_dict)
                        scheduler.walker_group.add_perf_data(indices_lst, results)
                        self.measurer.n_parallel = old_parallel
                        for indices, result in zip(indices_lst, results):
                            if result < float("inf"):
                                # if inf, maybe this measure is wrong
                                scheduler.walker_group.record(indices, result, random_reject=False)

                    scheduler.walker_group.clear_data()

                warm_up_epochs = 1
                warm_up_trials = self.measurer.n_parallel
                logger.info(f"warming up some more: epochs: {warm_up_epochs}, trials: {warm_up_trials}")
                res_dicts = self.warm_up(scheduler, warm_up_epochs, warm_up_trials, configs, type_keys, config_mode, use_model=use_model)
                result_dictionaries += res_dicts
            if scheduler.walker_group.top1_value() < minimal[1]:
                best = scheduler.walker_group.top1()
            else:
                best = minimal[0]

            return scheduler.walker_group.to_config(best), result_dictionaries

        elif mode == "manual":
            pass