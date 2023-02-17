from conductor.component.strategy._base import Strategy
from conductor.experiment import Experiment
from collections import OrderedDict
import numpy as np
import traceback

import logging
logger = logging.getLogger("conductor.component.strategy.trimmer")


# conductor-mechanism exeuction strategy (multiple operators) - formerly BFS
class TrimmerBatcher(object):
    _name = "trimmer_batcher"

    def __repr__(self):
        return TrimmerBatcher._name

    def __init__(self, _id, batch_size, trials_per_batcher):
        self.id = _id
        self.batch_size = batch_size
        self.trials_per_batcher = trials_per_batcher
        self.trials_remaining = self.trials_per_batcher
        self.should_stop = False
        self.curr_batch_id = 0
        self.curr_batch_size = 0
    
    def has_next(self):
        if self.should_stop:
            return False
        return self.trials_remaining > 0

    def next(self):
        retbid = self.curr_batch_id
        self.curr_batch_size = min(self.trials_remaining, self.batch_size)
        self.trials_remaining -= self.curr_batch_size
        self.curr_batch_id += 1
        return self.curr_batch_size, retbid

class TrimmerManager():
    _name = "trimmer_manager"

    def __repr__(self):
        return TrimmerManager._name

    def __init__(self, op_plat_thresh, mod_plat_thresh, window, cost_obj=None):
        self.op_plat_thresh = op_plat_thresh
        self.mod_plat_thresh = mod_plat_thresh
        self.window = window
        self.cost_obj = cost_obj
        self.cost = []
        self.plat = []
        self.hist_count = 0
        self.mod_plat = 0
        self.last_avg_improv = 1e15
        self.hist_improv = []
        self.batchers = OrderedDict()
    
    def add_batcher(self, bid, batcher):
        self.batchers[bid] = batcher
        self.cost.append(1e20)
        self.plat.append(0)

    def batcher(self, bid):
        return self.batchers[bid]

    def all_batchers(self):
        return list(self.batchers.values())

    def finished(self):
        count = 0
        for k, v in self.batchers.items():
            if not v.has_next():
                count += 1
        return count == len(self.batchers)
    
    # returns true if should stop, else false
    def step(self, total_cost):
        self.hist_count += 1
        self.hist_improv.append(total_cost)
        if len(self.hist_improv) > self.window:
            self.hist_improv.pop(0)
        elif len(self.hist_improv) < self.window:
            return False
        avg_improv = np.mean(self.hist_improv)
        if avg_improv > self.last_avg_improv:
            logger.info("current avg improvement(%s) > last avg improvement(%s)", str(avg_improv), str(self.last_avg_improv))
            if self.mod_plat > self.mod_plat_thresh:
                logger.info("model plateau reached threshold: %s", str(self.mod_plat_thresh))
                return True
            self.mod_plat += 1
            logger.info("increasing model plateau count to: %s", str(self.mod_plat))
        else:
            self.mod_plat = 0
            logger.info("current avg improvement(%s) < last avg improvement(%s), resetting model plateau, updating avg improvement", str(avg_improv), str(self.last_avg_improv))
            self.last_avg_improv = avg_improv
        if self.cost_obj and avg_improv <= self.cost_obj:
            logger.info("current avg improvement(%s) reached cost objective(%s)", str(avg_improv), str(self.cost_obj))
            return True
        return False

    def update_cost(self, bid, cost):
        if self.cost[bid] > cost:
            logger.info("new cost(%s) for batcher: %s is < current cost(%s), updating, resetting batcher plateau", str(cost), str(bid), str(self.cost[bid]))
            last_cost = self.cost[bid]
            last_plat = self.plat[bid]
            self.cost[bid] = cost
            self.plat[bid] = 0
        else:
            self.plat[bid] += 1
            logger.info("new cost(%s) for batcher: %s is > current cost(%s), increasing batcher plateau to: %s", str(cost), str(bid), str(self.cost[bid]), str(self.plat[bid]))
            if self.plat[bid] > self.op_plat_thresh:
                logger.info("batcher %s reached plateau threshold (%s), stopping batcher", str(bid), str(self.plat[bid]))
                self.batchers[bid].should_stop = True

class TrimmerStrategy(Strategy):
    _name = "trimmer"
    
    def __repr__(self):
        return Strategy.__repr__(self) + ":" + TrimmerStrategy._name

    def __init__(self, tasks_spec, measurer, builder, runner, evaluator, setting, results_path, configs=None, child_default_configs={}, profilers_specs=[]):
        Strategy.__init__(
            self, 
            "trimmer", 
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
        self.num_measures_per_round = 0
        self.num_measure_trials = 0

    def prepare_strategy(self, tasks):
        if "batch_size" in self.setting and self.setting["batch_size"] is not None and self.setting["batch_size"] >= 1:
            self.num_measures_per_round = self.setting["batch_size"]
        if "num_trials" not in self.setting or self.setting["num_trials"] is None:
            raise RuntimeError("num_trials should be specified for strategy and be at least 1")
        else:
            self.num_measure_trials = self.setting["num_trials"]
        if self.num_measures_per_round < 1:
            self.num_measures_per_round = 1
        if self.num_measure_trials < self.num_measures_per_round:
            self.num_measure_trials = self.num_measures_per_round
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
                    "configurer": None       
                }
        return res_dict
    
    def run(self):
        model, tasks = self.prepare_tasks()
        res_dict = self.prepare_strategy(tasks)
        
        manager = TrimmerManager(
            self.config.get("op_plateau_threshold", 2),
            self.config.get("mod_plateau_threshold", 2),
            self.config.get("window", 2)
        )

        for t in tasks:
            batcher = TrimmerBatcher(t["idx"], self.num_measures_per_round, self.num_measure_trials)
            manager.add_batcher(t["idx"], batcher)

        logger.info("start strategy proper")
        while not manager.finished():
            total_costs = []
            for tb in manager.all_batchers():
                logger.info("running batcher %s", str(tb.id))
                
                if not tb.has_next():
                    logger.info("batcher %s has no more trials left", str(tb.id))
                    continue
                else:
                    logger.info("batcher %s has trials left", str(tb.id))
                batch_size, index = tb.next()
                logger.info("batcher %s provided batch #%s, with size %s", str(tb.id), str(index), str(batch_size))
                _method, task, _, _, configurer, ctx = self.prepare_task(tasks[tb.id], "proper")
                self.profiling_checkpoint("task:start", ctx=ctx)
                Experiment.current.set_task(tb.id)
                Experiment.current.set_method(tb.id)
                logger.info("start task idx: %s:", str(tb.id))
                res_dict[str(tb.id)]["configurer"] = configurer
                self.measurer.set_stage("proper")
                self.measurer.set_task_idx(tb.id)
                _method.set_measurer(self.measurer)
                _method.load(task, configurer)
                logger.info("counter[%s], num_measure_trials[%s]", str(res_dict[str(tb.id)]["counter"]), str(self.num_measure_trials))

                try:
                    flop_dict, cost_dict, error_count, m_inputs, m_results = _method.execute(self.num_measures_per_round)
                    should_stop = self.should_stop_task(flop_dict, cost_dict, error_count, m_inputs, m_results)
                    res_dict[str(tb.id)]["total_inputs"] += m_inputs
                    res_dict[str(tb.id)]["total_results"] += m_results

                except Exception as e:
                    logger.error("stopping task due to exception!")
                    logger.error("exception" + traceback.format_exc())
                    should_stop = True

                if should_stop:
                    logger.info("task[%s] exhausted possible implementation candidates, moving on...", str(tb.id))
                    tb.should_stop = True
                else:
                    manager.update_cost(tb.id, cost_dict["cost"])
                    total_costs.append(cost_dict["cost"])
                
                _method.unload()

                self.profiling_checkpoint("task:stop")

            should_stop_forcefully = manager.step(sum(total_costs))
            logger.info("should_stop: %s", str(should_stop_forcefully))
            if should_stop_forcefully:
                break

        self.finalize(tasks, model, res_dict)
        logger.info("stop strategy proper")