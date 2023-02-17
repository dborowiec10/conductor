from conductor._base import Configurable
from conductor.profiler._base import Profilable
from conductor.configurer.configurer import Configurer
from conductor.component.measurer.doppler_base import DopplerBaseMeasurer
import time
import logging

from conductor.experiment import Experiment
logger = logging.getLogger("conductor.component.strategy")

class Strategy(Configurable, Profilable):
    _name = "strategy"
    
    def __repr__(self):
        return Strategy._name

    def __init__(self, subtype, tasks_spec, measurer, builder, runner, evaluator, setting, results_path, configs=None, child_default_configs={}, profilers_specs=[]):
        Configurable.__init__(self, "strategy", [subtype], configs, Configurable.merge_configs({}, child_default_configs, override_first=True))
        Profilable.__init__(self, "strategy", ["system_monitor", "time_profiler"], specs=profilers_specs)
        self.tasks_spec = tasks_spec
        self.measurer = measurer
        self.builder = builder
        self.runner = runner
        self.evaluator = evaluator
        self.setting = setting
        self.results_path = results_path
        self.configs = configs
        self.profilers_specs = profilers_specs

    def run(self):
        raise NotImplementedError()

    # implements strategy checkpointing for profilable
    def profiling_checkpoint(self, checkpoint, ctx=None):
        if checkpoint == "load":
            self.begin_profilers()

        elif checkpoint == "unload":
            self.end_profilers()

        elif checkpoint == "task:start":
            self.checkpoint_data["task:start"] = self.start_profilers(context=ctx)
            
        elif checkpoint == "task:stop":
            self.stop_profilers(self.checkpoint_data["task:start"])
            self.persist_profilers()

        else:
            pass

    def should_stop_task(self, flop_dict, cost_dict, error_count, m_inputs, m_results):
        if error_count != -1 and len(m_inputs) > 0:
            return False
        else:
            return True

    def prepare_tasks(self):
        model = {"model": None, "configurer": None}
        for k, t in enumerate(self.tasks_spec):
            if not model["model"] and t.get("model", None) is not None:
                model["model"] = t["model"]
                model["configurer"] = Configurer(self.results_path)
            if t["part_of_model"]:
                t["configurer"] = model["configurer"]
            else:
                t["configurer"] = Configurer(self.results_path)
        return model, self.tasks_spec

    def prepare_task(self, task_spec, strategy_stage):
        task_idx = task_spec["idx"]
        _method = task_spec["method"]
        task = task_spec["task"]
        task_weight = task_spec["task_weight"]
        configurer = task_spec["configurer"]
        return _method, task, task_idx, task_weight, configurer, {
            "strategy": self._subtypes[0],
            "strategy_setting": self.setting,
            "strategy_stage": strategy_stage,
            "measurer": str(self.measurer),
            "evaluator": str(self.evaluator),
            "builder": str(self.builder),
            "runner": str(self.runner)
        }

    def finalize(self, tasks, model, res_dict):
        if isinstance(self.measurer, DopplerBaseMeasurer):
            for t in tasks:
                _, task, task_idx, _, configurer, ctx = self.prepare_task(t, "rank")
                self.profiling_checkpoint("task:start", ctx=ctx)
                logger.info("strategy: rank remeasuring task: [%d]", task_idx)
                stime = time.time()
                self.measurer.measure_rank(
                    task, 
                    task_idx, 
                    res_dict[str(task_idx)]["total_inputs"], 
                    res_dict[str(task_idx)]["total_results"]
                )
                rank_duration = time.time() - stime
                Experiment.current.add_rank_duration(rank_duration)

                configurer.save_ranked_records()
                configurer.save_best_ranked_records()
                self.profiling_checkpoint("task:stop")

        if model["model"] is not None:
            model["configurer"].save_best_records()
            model["configurer"].save_all_records()
        else:
            for t in tasks:
                res_dict[str(t["idx"])]["configurer"].save_all_records()
                res_dict[str(t["idx"])]["configurer"].save_best_records()

    def load(self):
        self.profiling_checkpoint("load")
        self.evaluator = self.evaluator(configs=self.configs)
        self.builder = self.builder(configs=self.configs, profilers_specs=self.profilers_specs)
        self.runner = self.runner(self.evaluator, configs=self.configs, profilers_specs=self.profilers_specs)
        self.measurer = self.measurer(self.builder, self.runner, configs=self.configs)
        self.measurer.load()


    def unload(self):
        self.measurer.unload()
        self.profiling_checkpoint("unload")



