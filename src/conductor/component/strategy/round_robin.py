from conductor.component.strategy._base import Strategy
from conductor.experiment import Experiment
import numpy as np
import traceback

import logging
logger = logging.getLogger("conductor.component.strategy.round_robin")

class RoundRobinStrategy(Strategy):
    _name = "round_robin"

    def __repr__(self):
        return Strategy.__repr__(self) + ":" + RoundRobinStrategy._name

    def __init__(self, tasks_spec, measurer, builder, runner, evaluator, setting, results_path, configs=None, child_default_configs={}, profilers_specs=[]):
        Strategy.__init__(
            self, 
            "round_robin", 
            tasks_spec, 
            measurer, 
            builder, 
            runner, 
            evaluator, 
            setting, 
            results_path,
            configs=configs,
            child_default_configs=child_default_configs,
            profilers_specs=profilers_specs
        )
        self.num_warmup_sample = self.config.get("num_warmup_sample", 1)
        self.objective_func = sum
        self.early_stopping = 1e20
        self.num_measures_per_round = 0
        self.num_measure_trials = 0
        
        self.task_best_costs = []
        self.current_score = 0

        self.counter = 0
        self.best_counter = 0
        self.best_score = 0
        self.dead_tasks = set()

    def prepare_strategy(self, tasks):
        if "early_stop" in self.setting and self.setting["early_stop"] is not None and self.setting["early_stop"] >= 0:
            self.early_stopping = self.setting["early_stop"]
        else:
            self.early_stopping = 1e20
        if "batch_size" in self.setting and self.setting["batch_size"] is not None and self.setting["batch_size"] >= 1:
            self.num_measures_per_round = self.setting["batch_size"]
        else:
            raise RuntimeError("batch_size should be specified for strategy and be at least 1")
        if "num_trials" not in self.setting or self.setting["num_trials"] is None:
            raise RuntimeError("num_trials should be specified for strategy and be at least 1")
        else:
            self.num_measure_trials = self.setting["num_trials"]
        if self.num_measure_trials * len(tasks) < self.num_measures_per_round * len(tasks):
            self.num_measure_trials = self.num_measures_per_round
        self.num_measure_trials = self.num_measure_trials * len(tasks)
        self.num_measure_trials = self.num_measure_trials + (self.num_warmup_sample * len(tasks))
        self.task_best_costs = 1e10 * np.ones(len(tasks))
        res_dict = {}
        for t in tasks:
            if str(t["idx"]) not in res_dict:
                res_dict[str(t["idx"])] = {
                    "best_flops": 0,
                    "best_config": None,
                    "best_pair": None,
                    "best_idx": 0,
                    "counter": 0,
                    "total_errors": 0,
                    "total_results": [],
                    "total_inputs": [],
                    "configurer": None,                 
                }
        return res_dict

    def run(self):
        model, tasks = self.prepare_tasks()
        res_dict = self.prepare_strategy(tasks)
        idx = -1

        logger.info("start strategy proper")
        while self.counter < self.num_measure_trials and len(self.dead_tasks) < len(tasks):
            self.profiling_checkpoint("task:start", ctx=ctx)

            logger.info("counter[%s], num_measure_trials[%s], len_dead_tasks[%s], len_tasks[%s]",
                str(self.counter),
                str(self.num_measure_trials),
                str(len(self.dead_tasks)),
                str(len(tasks))
            )

            # choose index
            idx = (idx + 1) % len(tasks)
            while idx in self.dead_tasks:
                idx = (idx + 1) % len(tasks)
            logger.info("counter: %s, next task idx %s", str(self.counter), str(idx))

            _method, task, _, _, configurer, ctx = self.prepare_task(tasks[idx], "proper")
            
            res_dict[str(idx)]["configurer"] = configurer

            Experiment.current.set_task(idx)
            Experiment.current.set_method(idx)
        
            self.measurer.set_stage("proper")
            self.measurer.set_task_idx(idx)

            _method.set_measurer(self.measurer)
            _method.load(task, configurer)

            try:
                flop_dict, cost_dict, error_count, m_inputs, m_results = _method.execute(self.num_measures_per_round)
                should_stop = self.should_stop_task(flop_dict, cost_dict, error_count, m_inputs, m_results)
                res_dict[str(idx)]["total_inputs"] += list(m_inputs)
                res_dict[str(idx)]["total_results"] += list(m_results)

            except Exception:
                logger.error("stopping task due to exception!")
                logger.error("exception" + traceback.format_exc())
                should_stop = True
            
            if should_stop:
                logger.info("task[%s] adding to dead tasks list", str(idx))
                self.dead_tasks.add(idx)
            else:
                prev_best = self.task_best_costs[idx]
                if cost_dict["cost"] < self.task_best_costs[idx]:
                    self.task_best_costs[idx] = cost_dict["cost"]
                    logger.info("task[%s] prev_best_cost: %s, current_best_cost: %s", str(idx), str(prev_best), str(self.task_best_costs[idx]))
                else:
                    logger.info("task[%s] prev_best_cost: %s, current_best_cost: %s", str(idx), str(prev_best), str(prev_best))

                prev_counter = self.counter
                prev_score = self.current_score

                self.counter += len(m_results)
                self.current_score = self.objective_func(self.task_best_costs)

                logger.info("prev counter: %s, current counter %s, best_counter: %s", str(prev_counter), str(self.counter), str(self.best_counter))
                logger.info("prev score: %s, current score %s, best_score: %s", str(prev_score), str(self.current_score), str(self.best_score))
             
            _method.unload()

            self.profiling_checkpoint("task:stop")

            if not self.best_score:
                self.best_counter = self.counter
                self.best_score = self.current_score
                logger.info("counter: %s, best_counter: %s, cur_score: %s, best_score: %s",
                    str(self.counter),
                    str(self.best_counter),
                    str(self.current_score),
                    str(self.best_score)
                )
            else:
                if self.current_score < self.best_score:
                    self.best_score = self.current_score
                    self.best_counter = self.counter
                    logger.info("counter: %s, best_counter: %s, cur_score: %s, best_score: %s",
                        str(self.counter),
                        str(self.best_counter),
                        str(self.current_score),
                        str(self.best_score)
                    )
                elif self.counter - self.best_counter >= self.early_stopping and all(cost < 1e9 for cost in self.task_best_costs):
                    logger.info("early stopping, no improvement in the last %s trials", str(self.early_stopping))
                    break

        self.finalize(tasks, model, res_dict)
        logger.info("stop strategy proper")