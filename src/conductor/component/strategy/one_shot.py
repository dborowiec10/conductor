from conductor.component.strategy._base import Strategy
from conductor.experiment import Experiment
import traceback

import logging
logger = logging.getLogger("conductor.component.strategy.one_shot")

class OneShotStrategy(Strategy):
    _name = "one_shot"
    
    def __repr__(self):
        return Strategy.__repr__(self) + ":" + OneShotStrategy._name

    def __init__(self, tasks_spec, measurer, builder, runner, evaluator, setting, results_path, configs=None, child_default_configs={}, profilers_specs=[]):
        Strategy.__init__(
            self, 
            "one_shot", 
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
        self.num_measure_trials = 0

    def prepare_strategy(self, tasks):
        if "num_trials" not in self.setting or self.setting["num_trials"] is None:
            raise RuntimeError("num_trials should be specified for strategy and be at least 1")
        else:
            self.num_measure_trials = self.setting["num_trials"]
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

        logger.info("start strategy proper")

        for t in tasks:
            _method, task, task_idx, _, configurer, ctx = self.prepare_task(t, "proper")

            self.profiling_checkpoint("task:start", ctx=ctx)

            Experiment.current.set_task(task_idx)
            Experiment.current.set_method(task_idx)
            logger.info("start task idx: %s:", str(task_idx))
            res_dict[str(task_idx)]["configurer"] = configurer

            self.measurer.set_stage("proper")
            self.measurer.set_task_idx(task_idx)

            _method.set_measurer(self.measurer)
            _method.load(task, configurer)

            _best_flops = 0
            _best_config = None
            _best_pair = None
            _best_idx = 0
            i = 0
            total_error_count = 0
            logger.info("counter[0], num_measure_trials[%s]", str(self.num_measure_trials))
            try:
                flop_dict, cost_dict, error_count, m_inputs, m_results = _method.execute(self.num_measure_trials)
                res_dict[str(task_idx)]["total_inputs"] += m_inputs
                res_dict[str(task_idx)]["total_results"] += m_results

            except Exception as e:
                logger.error("stopping task due to exception!")
                logger.error("exception" + traceback.format_exc())

            if flop_dict["flop"] > _best_flops:
                logger.info("updating, prev flops: %s, current flops: %s", str(_best_flops), str(flop_dict["flop"]))
                logger.info("updating, prev config: %s, current config: %s", str(_best_config), str(flop_dict["config"]))
                logger.info("updating, prev pair: %s, current pair: %s", str(_best_pair), str(flop_dict["pair"]))
                logger.info("updating, prev idx: %s, current idx: %s", str(_best_idx), str(i + flop_dict["idx"]))

                _best_flops = flop_dict["flop"]
                _best_config = flop_dict["config"]
                _best_pair = flop_dict["pair"]
                _best_idx = i + flop_dict["idx"]

            total_error_count += error_count

            if total_error_count >= int(0.75 * float(self.num_measure_trials)):
                logger.debug("More than 75% of candidates errored out!")

            _method.unload()

            logger.info("stop task idx: %s:", str(task_idx))

            self.profiling_checkpoint("task:stop")
            
        self.finalize(tasks, model, res_dict)
        logger.info("stop strategy proper")