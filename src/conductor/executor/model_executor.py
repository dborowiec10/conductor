from conductor.job.task import Task
from conductor._base import Configurable
from conductor.profiler._base import Profilable
from conductor.executor.model_executor_routine import execute_routine
from conductor.worker.worker import ExecutingWorker, StatusKind
from conductor.mediation import ERROR_TYPES, MeasureErrorNo

import logging

from conductor.experiment import Experiment

logger = logging.getLogger("conductor.compiler.model_executor")

class ModelExecutor(Task, Configurable, Profilable):
    _name = "model_executor"
    
    def __repr__(self):
        return Task.__repr__(self) + ":" + ModelExecutor._name

    def __init__(self, spec, task_results_path, models_path, tensor_programs_path):
        Task.__init__(
            self, 
            "model",
            spec,
            ["compiled_model"],
            [],
            ["execution"],
            task_results_path,
            models_path,
            tensor_programs_path
        )
        Configurable.__init__(self, "executor", ["model"], self.spec.configurations, Configurable.merge_configs({
            "timeout": 20
        }, {}, override_first=True))

        Profilable.__init__(self, "executor", ["time_profiler", "system_monitor"], specs=self.spec.profilers)

        self.specs = self.find_specifications_by_type("execution")
        self.timeout = self.config.get("timeout", 20)
        self.error_map = {k:i for k, i in enumerate(ERROR_TYPES)}

    # implements strategy checkpointing for profilable
    def profiling_checkpoint(self, checkpoint, ctx={}):
        if checkpoint == "begin":
            self.begin_profilers()
        elif checkpoint == "end":
            self.persist_profilers()
            self.end_profilers()
        if checkpoint == "start":
            self.checkpoint_data["task:start"] = self.start_profilers(context=ctx)
        elif checkpoint == "stop":
            self.append_context_profilers(self.checkpoint_data["task:start"], ctx)
            self.stop_profilers(self.checkpoint_data["task:start"])
        else:
            pass

    def get_routine(self):
        return execute_routine

    def callback(self, retval):
        status, _, res = retval
        if status not in [StatusKind.COMPLETE, None]:
            error_no = MeasureErrorNo.RUN_TIMEOUT if status == StatusKind.TIMEOUT else MeasureErrorNo.RUNTIME_DEVICE
        else:
            error_no = MeasureErrorNo.NO_ERROR
            
        if res is not None:
            out, error_msg = res
            if error_msg is not None:
                error_no = MeasureErrorNo.RUNTIME_DEVICE
        else:
            out = (None, None, None, None)

        costs, mean, total_time, other = out

        self.profiling_checkpoint("run:stop", ctx={
            "error_no": self.error_map[error_no],
            "costs": costs,
            "mean": mean,
            "total_time": total_time,
            "other": other
        })

    def run(self, inputs, idx):
        Experiment.current.set_config({
            "timeout": self.config.get("timeout", 20)
        })
        stat = "Executing Models: ["
        inp = self.prepare_inputs(inputs)
        pairs = []
        for s in self.specs:
            pairs += s.pair_inputs_options(inp, self.spec.configurations)

        self.profiling_checkpoint("begin")

        results = []
        statuses = []
        worker = ExecutingWorker()
        left_to_do = len(pairs)
        glob_idx = 0

        while left_to_do > 0:
            mod, opts, evaluator = pairs[glob_idx]

            Experiment.current.add_md_and_opt(
                {
                    "idx": glob_idx,
                    "name": mod.model_name,
                    "framework": mod.model_framework,
                    "batch_size": mod.model_batch_size,
                    "in_shape": mod.model_in_shape,
                    "out_shape": mod.model_out_shape, 
                    "opt_level": mod.opt_level,
                    "instruction_count": mod.instr_count,
                    "implementations_type": str(mod.implementations_type),
                    "implementations_path": str(mod.implementations_path),
                },
                {
                    "device_type": opts.device_type,
                    "device_id": str(opts.device_id),
                    "fill_mode": opts.fill_mode,
                    "evaluator": {**{"type": evaluator._type}, **evaluator.config}
                }
            )

            Experiment.current.set_index(glob_idx)

            self.profiling_checkpoint("start")

            worker.submit(
                self.get_routine(),
                (mod, opts, evaluator),
                timeout=self.timeout,
                callback=self.callback
            )
            status, _, res = worker.get()
            if status != StatusKind.COMPLETE:
                result = None
                if status == StatusKind.TIMEOUT:
                    exec_stat = "T"
                else:
                    exec_stat = "E"
            else:
                _res, error_msg = res
                if error_msg is not None:
                    exec_stat = "E"
                else:
                    exec_stat = "*"
                result = _res
                
            results.append(result)
            statuses.append(exec_stat)
            left_to_do -= 1
            glob_idx += 1

        ExecutingWorker.kill_all([worker])
        self.profiling_checkpoint("end")

        stat += "".join(statuses)
        stat += "]"
        logger.info(stat)
        return self.prepapre_outputs(results)