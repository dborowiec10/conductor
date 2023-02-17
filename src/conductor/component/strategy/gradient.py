from conductor.component.strategy._base import Strategy
from conductor.experiment import Experiment
from conductor.mediation import SketchTask, Tasker

import traceback
import math
import numpy as np

import logging
logger = logging.getLogger("conductor.component.strategy.gradient")

class GradientStrategy(Strategy):
    _name = "gradient"
    
    def __repr__(self):
        return Strategy.__repr__(self) + ":" + GradientStrategy._name

    def __init__(self, tasks_spec, measurer, builder, runner, evaluator, setting, results_path, configs=None, child_default_configs={}, profilers_specs=[]):
        Strategy.__init__(
            self, 
            "gradient",
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

        self.alpha = self.config.get("alpha", 0.2)
        self.beta = self.config.get("beta", 2)
        self.backward_window_size = self.config.get("backward_window_size", 3)
        self.num_warmup_sample = self.config.get("num_warmup_sample", 1)
        self.objective_func = sum
        self.early_stopping = 1e20
        self.num_measures_per_round = 0
        self.num_measure_trials = 0

        self.task_counters = []
        self.task_costs_history = []
        self.task_best_costs = []
        self.current_score = 0

        self.flop_counts = []
        self.task_tags = []
        self.tag_to_group_id = {}
        self.group_task_ids = []

        self.counter = 0
        self.best_counter = 0
        self.best_score = 0
        self.dead_tasks = set()

    def get_similarity_tag(self, task):
        ret = ""
        flop = Tasker.task_theoretical_flop(task)
        if isinstance(task, SketchTask):
            for op in task.compute_dag.ops:
                tag = op.attrs.get("auto_scheduler_task_scheduler_tag", None)
                if tag:
                    ret += op.attrs["auto_scheduler_task_scheduler_tag"] + "_"
        if ret:
            ret += "%d" % int(math.log(flop + 1, 1.618))
        return ret, flop

    def adjust_similarity_group(self, task_idx):
        group_id = self.tag_to_group_id.get(self.task_tags[task_idx], None)
        if group_id is None or len(self.group_task_ids[group_id]) <= 1:
            return
        group_ids = self.group_task_ids[group_id]
        best_group_flops = max([self.flop_counts[j] / self.task_best_costs[j] for j in group_ids])
        cur_flops = self.flop_counts[task_idx] / self.task_best_costs[task_idx]
        if cur_flops < best_group_flops / self.beta and self.task_counters[task_idx] > 5 + max(self.task_counters[j] for j in group_ids if j != task_idx):
            self.task_tags[task_idx] = None
            group_ids.remove(task_idx)

    def prepare_strategy(self, tasks):
        res_dict = {}
        if self.config.get("objective_func", "") == "weighted_sum":
            weights = [t["details"]["task_weight"] for t in tasks]
            self.objective_func = lambda costs: sum(c * w for c, w in zip(costs, weights))
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
        self.task_counters = [0 for _ in range(len(tasks))]
        self.task_costs_history = [[] for _ in range(len(tasks))]
        self.task_best_costs = 1e10 * np.ones(len(tasks))
        self.current_score = self.objective_func(self.task_best_costs)
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
            tag, flop = self.get_similarity_tag(t["task"])
            self.task_tags.append(tag)
            self.flop_counts.append(flop)
            if not tag:
                continue
            if tag not in self.tag_to_group_id:
                self.tag_to_group_id[tag] = len(self.tag_to_group_id)
                self.group_task_ids.append([])
            self.group_task_ids[self.tag_to_group_id[tag]].append(t["idx"])
        return res_dict
        
    def round_robin_warmup(self, tasks, warmup_cnt, res_dict):
        for t in tasks:
            if not self.task_counters[t["idx"]]:
                _method, task, _, _, configurer, ctx = self.prepare_task(tasks[t["idx"]], "warm-up")
                self.profiling_checkpoint("task:start", ctx=ctx)
                Experiment.current.set_task(t["idx"])
                Experiment.current.set_method(t["idx"])
                logger.info("round-robin warmup, start task idx: %s", str(t["idx"]))

                res_dict[str(t["idx"])]["configurer"] = configurer

                self.measurer.set_stage("warmup")
                self.measurer.set_task_idx(t["idx"])

                _method.set_measurer(self.measurer)
                _method.load(task, configurer)

                try:
                    flop_dict, cost_dict, error_count, m_inputs, m_results = _method.execute(warmup_cnt)
                    should_stop = self.should_stop_task(flop_dict, cost_dict, error_count, m_inputs, m_results)
                    res_dict[str(t["idx"])]["total_inputs"] += list(m_inputs)
                    res_dict[str(t["idx"])]["total_results"] += list(m_results)

                except Exception:
                    logger.error("round-robin warmup, stopping task due to exception!")
                    logger.error("exception" + traceback.format_exc())
                    should_stop = True

                if not should_stop:
                    if cost_dict["cost"] < self.task_best_costs[t["idx"]]:
                        self.task_best_costs[t["idx"]] = cost_dict["cost"]
                    if len(m_inputs) == 0:
                        self.dead_tasks.add(t["idx"])
                    self.task_counters[t["idx"]] += 1
                    self.task_costs_history[t["idx"]].append(self.task_best_costs[t["idx"]])
                    self.counter += len(m_results)
                    self.current_score = self.objective_func(self.task_best_costs)
                
                _method.unload()

                logger.info("round-robin warmup, stop task idx: %s", str(t["idx"]))

                self.profiling_checkpoint("task:stop")

        self.best_counter = self.counter
        self.best_score = self.current_score

        logger.info("round-robin warmup, counter: %s, best_counter: %s, cur_score: %s, best_score: %s",
            str(self.counter),
            str(self.best_counter),
            str(self.current_score),
            str(self.best_score)
        )
        logger.info("round-robin warmup, task_best_costs: [%s]", ",".join([str(tt) for tt in self.task_best_costs]))
        logger.info("round-robin warmup, dead_tasks: [%s]", ",".join([str(tt) for tt in self.dead_tasks]))
        logger.info("round-robin warmup, task_counters: [%s]", ",".join([str(tt) for tt in self.task_counters]))

        return res_dict

    def grad_suggest_idx(self, tasks):
        gradients = []
        for i in range(len(tasks)):
            if i in self.dead_tasks:
                gradients.append(0)
                continue

            delta = 1e-4
            new_costs = list(self.task_best_costs)
            new_costs[i] -= delta
            chain_grad = (self.objective_func(self.task_best_costs) - self.objective_func(new_costs)) / delta
            if (self.task_counters[i] - 1 < len(self.task_costs_history[i])) and (self.task_counters[i] - 1 - self.backward_window_size >= 0):
                backward_grad = (
                            self.task_costs_history[i][self.task_counters[i] - 1] - self.task_costs_history[i][self.task_counters[i] - 1 - self.backward_window_size]
                        ) / self.backward_window_size
            else:
                backward_grad = 0
            
            g_next_1 = self.task_best_costs[i] - (self.task_best_costs[i] / self.task_counters[i])
            g_next_2 = self.beta * 1e30
            group_id = self.tag_to_group_id.get(self.task_tags[i], None)
            if group_id is not None and len(self.group_task_ids[group_id]) > 1:
                best_flops = max([
                    self.flop_counts[j] / self.task_best_costs[j]
                    for j in self.group_task_ids[group_id]
                ])
                g_next_2 = self.beta * self.flop_counts[i] / best_flops
            g_next = min(g_next_1, g_next_2)
            forward_grad = g_next - self.task_best_costs[i]
            # combine all grads
            grad = chain_grad * (self.alpha * backward_grad + (1 - self.alpha) * forward_grad)
            assert grad <= 0
            gradients.append(grad)

        logger.info("gradient suggest idx, gradients: [%s]", ",".join([str(g) for g in gradients]))
        
        if max(gradients) == min(gradients):
            task_idx = np.random.choice(len(gradients))
        else:
            task_idx = np.argmin(gradients)

        return task_idx

    def run(self):
        model, tasks = self.prepare_tasks()
        res_dict = self.prepare_strategy(tasks)

        logger.info("start round-robin warmup")
        res_dict = self.round_robin_warmup(tasks, self.num_warmup_sample, res_dict)
        logger.info("stop round-robin warmup")

        logger.info("start strategy proper")
        while self.counter < self.num_measure_trials and len(self.dead_tasks) < len(tasks):
            logger.info("counter[%s], num_measure_trials[%s], len_dead_tasks[%s], len_tasks[%s]",
                str(self.counter),
                str(self.num_measure_trials),
                str(len(self.dead_tasks)),
                str(len(tasks))
            )
            idx = self.grad_suggest_idx(tasks)
            logger.info("counter: %s, next task idx %s", str(self.counter), str(idx))
            _method, task, _, _, configurer, ctx = self.prepare_task(tasks[idx], "proper")

            self.profiling_checkpoint("task:start", ctx=ctx)

            Experiment.current.set_task(idx)
            Experiment.current.set_method(idx)
            res_dict[str(idx)]["configurer"] = configurer

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

                self.task_counters[idx] += 1
                self.task_costs_history[idx].append(self.task_best_costs[idx])

                prev_counter = self.counter
                prev_score = self.current_score

                self.counter += len(m_results)
                self.current_score = self.objective_func(self.task_best_costs)

                logger.info("prev counter: %s, current counter %s, best_counter: %s", str(prev_counter), str(self.counter), str(self.best_counter))
                logger.info("prev score: %s, current score %s, best_score: %s", str(prev_score), str(self.current_score), str(self.best_score))

                self.adjust_similarity_group(idx)
  
            _method.unload()

            self.profiling_checkpoint("task:stop")

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

