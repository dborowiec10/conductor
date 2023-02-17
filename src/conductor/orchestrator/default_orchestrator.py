from conductor.orchestrator._base import Orchestrator
from conductor.mediation import Tasker
from conductor.component.scheduler.scheduler import schedulers
from conductor.experiment import Experiment
from conductor.utils import get_const_int

import json

import logging
logger = logging.getLogger("conductor.orchestrator")

class DefaultOrchestrator(Orchestrator):
    _name = "default"

    def __repr__(self):
        return Orchestrator.__repr__(self) + ":" + DefaultOrchestrator._name

    def __init__(self, spec, task_results_path, models_path, tensor_programs_path):
        Orchestrator.__init__(
            self,
            "default", 
            spec, 
            ["model", "tensor_program"],
            [],
            ["orchestrator", "method", "input_method_map"],
            task_results_path,
            models_path, 
            tensor_programs_path
        )

    def prepare_method(self, method_spec, len_tasks, cost_model_override=None):
        sched_cls = schedulers[method_spec.scheduling]
        if method_spec.kind == "composite":
            if method_spec.scheduling == "template":
                method = method_spec.method(
                    sched_cls,
                    method_spec.cost_model,
                    method_spec.optimizer,
                    method_spec.sampler,
                    method_spec.filter,
                    configs=self.spec.configurations,
                    profilers_specs=self.spec.profilers
                )
            elif method_spec.scheduling == "sketch":
                if not cost_model_override:
                    cm = method_spec.cost_model
                    if "num_warmup_samples" in self.orch_spec.settings:
                        num_warmup = self.orch_spec.settings["num_warmup_samples"]
                    else:
                        num_warmup = min(self.orch_spec.settings["batch_size"], self.orch_spec.settings["num_trials"] / len_tasks)
                    cm = cm(num_warmup, configs=self.spec.configurations)
                else:
                    cm = cost_model_override

                method = method_spec.method(
                    sched_cls,
                    cm,
                    method_spec.search_policy,
                    configs=self.spec.configurations,
                    profilers_specs=self.spec.profilers
                )
        elif method_spec.kind == "standalone":
            method = method_spec.method(
                sched_cls, 
                configs=self.spec.configurations, 
                profilers_specs=self.spec.profilers
            )
        return method

    def prepare_model_plan(self, map):
        strategy = map["strategy"]
        model = map["pairs"][0][2]
        method_spec = map["pairs"][0][3]
        mod, params = (model.get_json(), model.get_params())
        strategy_inputs = []

        measurer_config = self.find_configuration_by_entity("measurer:" + self.orch_spec.measurer_spec)
        target = measurer_config.c["target"]
        target_host = measurer_config.c["target_host"]

        extracted_tasks = Tasker.extract_tasks(mod, params, model.model_name, target, target_host)
        if method_spec.scheduling in ["template"]:
            extracted_tasks = [ex for ex in extracted_tasks if ex["kind"] == "templateable"]
        else:
            extracted_tasks = [ex for ex in extracted_tasks if ex["kind"] == "untemplateable"]

        if method_spec.scheduling == "sketch":
            cm = method_spec.cost_model
            if "num_warmup_samples" in self.orch_spec.settings:
                num_warmup = self.orch_spec.settings["num_warmup_samples"]
            else:
                num_warmup = min(self.orch_spec.settings["batch_size"], self.orch_spec.settings["num_trials"] / len(extracted_tasks))
            cm = cm(num_warmup, configs=self.spec.configurations)
        else:
            cm = None

        for k, ex in enumerate(extracted_tasks):
            logger.debug(
                "extracted tensor program (%s): [%s]: [%s], weight: [%s], tasks: [%s]",
                str(k),
                ex["kind"],
                ex["def"]["workload_key"],
                ex["def"]["weight"],
                str([Tasker.task_repr(t) for t in ex["tasks"]])
            )

            method = self.prepare_method(method_spec, len(map["pairs"]), cost_model_override=cm)
            task = ex["tasks"][method_spec.scheduling]
            logger.info("prepared task-->method: task[%s] --> method[%s]", Tasker.task_repr(task), str(method))

            strategy_inputs.append({
                "idx": k,
                "task": task,
                "task_weight": ex["def"]["weight"],
                "part_of_model": True,
                "method": method,
                "model": model,
            })
            
            Experiment.current.add_task({
                "scheduling_type": Tasker.task_type(task),
                "weight": get_const_int(ex["def"]["weight"]),
                "name": getattr(task, "identifier", None),
                "func_name": getattr(task, "func_name", None),
                "args": Tasker.task_args_repr(task),
                "flop": Tasker.task_theoretical_flop(task),
                "target": str(task.target),
                "target_host": str(task.target_host),
                "part_of_model": True
            }, k)

            Experiment.current.add_method({
                "kind": method_spec.kind,
                "scheduling": method_spec.scheduling,
                "name": method_spec.method_name,
                "cost_model": method_spec.cost_model_spec,
                "optimizer": method_spec.optimizer_spec,
                "search_policy": method_spec.search_policy_spec,
                "sampler": method_spec.sampler_spec,
                "filter": method_spec.filter_spec
            }, k)

        strategy = strategy(
            strategy_inputs, 
            self.orch_spec.measurer,
            self.orch_spec.builder,
            self.orch_spec.runner,
            self.orch_spec.evaluator,
            self.orch_spec.settings,
            self.results_path,
            configs=self.spec.configurations,
            profilers_specs=self.spec.profilers
        )

        logger.info("prepared strategy: %s", str(strategy))
        return strategy

    def prepare_tensor_program_plan(self, map):
        strategy = map["strategy"]
        strategy_inputs = []
        for _, _, inp, method_spec in map["pairs"]:
            method = self.prepare_method(method_spec, len(map["pairs"]))
            task = Tasker.task_from_tensor_program(inp, method_spec.scheduling)
            cur_idx = len(strategy_inputs)

            strategy_inputs.append({
                "idx": cur_idx,
                "task": task,
                "task_weight": 1,
                "part_of_model": False,
                "method": method,
            })

            Experiment.current.add_task({
                "scheduling_type": Tasker.task_type(task),
                "weight": 1,
                "name": getattr(task, "identifier", None),
                "func_name": getattr(task, "func_name", None),
                "args": Tasker.task_args_repr(task),
                "flop": Tasker.task_theoretical_flop(task),
                "target": str(task.target),
                "target_host": str(task.target_host),
                "part_of_model": False
            }, cur_idx)

            Experiment.current.add_method({
                "kind": method_spec.kind,
                "scheduling": method_spec.scheduling,
                "name": method_spec.method_name,
                "cost_model": method_spec.cost_model_spec,
                "optimizer": method_spec.optimizer_spec,
                "search_policy": method_spec.search_policy_spec,
                "sampler": method_spec.sampler_spec,
                "filter": method_spec.filter_spec
            }, cur_idx)
        
        strategy = strategy(
            strategy_inputs, 
            self.orch_spec.measurer,
            self.orch_spec.builder,
            self.orch_spec.runner,
            self.orch_spec.evaluator,
            self.orch_spec.settings,
            self.results_path,
            configs=self.spec.configurations,
            profilers_specs=self.spec.profilers
        )

        setting = self.orch_spec.settings.copy()
        setting.update(strategy.config)

        Experiment.current.set_strategy(
            map["strategy_spec"],
            setting,
            self.orch_spec.measurer_spec,
            self.orch_spec.evaluator_spec,
            self.orch_spec.builder_spec,
            self.orch_spec.runner_spec
        )

        logger.info("prepared strategy: %s", str(strategy))

        return strategy

    def prepare_execution_plans(self, inputs):
        method_specs = self.find_specifications_by_type("method")
        imms = self.find_specifications_by_type("input_method_map")
        for i in inputs.values():
            i._load()

        all_model_maps = []
        all_tensor_program_maps = []
        for i in imms:
            model_maps, tensor_program_maps = i.pair_inputs_methods(inputs, method_specs)
            all_model_maps += model_maps
            all_tensor_program_maps += tensor_program_maps

        tuning_plans = []
        for amm in all_model_maps:
            tuning_plans.append(self.prepare_model_plan(amm))

        for tpmm in all_tensor_program_maps:
            tuning_plans.append(self.prepare_tensor_program_plan(tpmm))

        return tuning_plans

    def run(self, inputs, idx):
        inp = self.prepare_inputs(inputs)
        strategies = self.prepare_execution_plans(inp)
        for strategy in strategies:
            strategy.load()
            logger.info("start strategy execution: %s", str(strategy))
            strategy.run()
            logger.info("stop strategy execution: %s", str(strategy))
            strategy.unload()

        out = []
        return self.prepapre_outputs(out)