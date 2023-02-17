from conductor.component.strategy._base import Strategy
from conductor.experiment import Experiment
import traceback
import logging

logger = logging.getLogger("conductor.component.strategy.sequential")

class SequentialStrategy(Strategy):
    _name = "sequential"
    
    def __repr__(self):
        return Strategy.__repr__(self) + ":" + SequentialStrategy._name

    def __init__(self, tasks_spec, measurer, builder, runner, evaluator, setting, results_path, configs=None, child_default_configs={}, profilers_specs=[]):
        Strategy.__init__(
            self, 
            "sequential", 
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
        if "early_stop" in self.setting and self.setting["early_stop"] is not None and self.setting["early_stop"] >= 0:
            self.early_stopping = self.setting["early_stop"]
        else:
            self.early_stopping = 1e20
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
                    "configurer": None,                 
                }    
        return res_dict


    def run(self):
        model, tasks = self.prepare_tasks()
        res_dict = self.prepare_strategy(tasks)
        logger.info("start strategy proper")

        for t in tasks:
            _method, task, task_idx, _, configurer, ctx = self.prepare_task(t, "proper")
            logger.info("start task idx: %s:", str(task_idx))

            self.profiling_checkpoint("task:start", ctx=ctx)

            res_dict[str(task_idx)]["configurer"] = configurer

            Experiment.current.set_task(task_idx)
            Experiment.current.set_method(task_idx)
            
            self.measurer.set_stage("proper")
            self.measurer.set_task_idx(task_idx)
            
            _method.set_measurer(self.measurer)
            _method.load(task, configurer)

            while res_dict[str(task_idx)]["counter"] < self.num_measure_trials:
                logger.info("counter[%s], num_measure_trials[%s]", str(res_dict[str(task_idx)]["counter"]), str(self.num_measure_trials))
                
                try:
                    flop_dict, cost_dict, error_count, m_inputs, m_results = _method.execute(self.num_measures_per_round)
                    should_stop = self.should_stop_task(flop_dict, cost_dict, error_count, m_inputs, m_results)
                    res_dict[str(task_idx)]["total_inputs"] += m_inputs
                    res_dict[str(task_idx)]["total_results"] += m_results

                except Exception as e:
                    logger.error("stopping task due to exception!")
                    logger.error("exception" + traceback.format_exc())
                    should_stop = True

                if should_stop:
                    logger.info("task[%s] exhausted possible implementation candidates, moving on...", str(task_idx))
                    break

                if flop_dict["flop"] > res_dict[str(task_idx)]["best_flops"]:
                    logger.info("updating, prev flops: %s, current flops: %s", str(res_dict[str(task_idx)]["best_flops"]), str(flop_dict["flop"]))
                    logger.info("updating, prev config: %s, current config: %s", str(res_dict[str(task_idx)]["best_config"]), str(flop_dict["config"]))
                    logger.info("updating, prev pair: %s, current pair: %s", str(res_dict[str(task_idx)]["best_pair"]), str(flop_dict["pair"]))
                    logger.info("updating, prev idx: %s, current idx: %s", str(res_dict[str(task_idx)]["best_idx"]), str(res_dict[str(task_idx)]["counter"] + flop_dict["idx"]))
                    res_dict[str(task_idx)]["best_flops"] = flop_dict["flop"]
                    res_dict[str(task_idx)]["best_config"] = flop_dict["config"]
                    res_dict[str(task_idx)]["best_pair"] = flop_dict["pair"]
                    res_dict[str(task_idx)]["best_idx"] = res_dict[str(task_idx)]["counter"] + flop_dict["idx"]

                res_dict[str(task_idx)]["total_errors"] += error_count
                res_dict[str(task_idx)]["counter"] += len(m_results)
                if res_dict[str(task_idx)]["total_errors"] >= int(0.75 * float(self.num_measure_trials)):
                    logger.debug("More than 75% of candidates errored out!")

                if res_dict[str(task_idx)]["counter"] > res_dict[str(task_idx)]["best_idx"] + self.early_stopping:
                    logger.info("early stopping, no improvement in the last %s trials. Stopping at idx: %s", str(self.early_stopping), res_dict[str(task_idx)]["best_idx"])
                    break
            _method.unload()

            self.profiling_checkpoint("task:stop")

            logger.info("stop task idx: %s:", str(task_idx))
        
        self.finalize(tasks, model, res_dict)
        logger.info("stop strategy proper")