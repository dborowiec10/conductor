from tabnanny import check
from conductor.job.task import Task
from conductor.mediation import MeasureErrorNo
from conductor._base import Configurable
from conductor.worker.worker import ExecutingWorker, StatusKind
from conductor.compiler.tensor_program_compiler_routine import compile_routine
from conductor.profiler._base import Profilable
from conductor.mediation import ERROR_TYPES, MeasureErrorNo

import psutil

import logging

from conductor.experiment import Experiment

logger = logging.getLogger("conductor.compiler.tensor_program_compiler")

class TensorProgramCompiler(Task, Configurable, Profilable):
    _name = "tensor_program_compiler"
    
    def __repr__(self):
        return Task.__repr__(self) + ":" + TensorProgramCompiler._name

    def __init__(self, spec, task_results_path, models_path, tensor_programs_path):
        Task.__init__(
            self, 
            "tensor_program",
            spec, 
            ["tensor_program"],
            [],
            ["compilation"],
            task_results_path,
            models_path, 
            tensor_programs_path
        )
        Configurable.__init__(self, "compiler", ["tensor_program"], self.spec.configurations, Configurable.merge_configs({
            "n_parallel": None,
            "timeout": 20
        }, {}, override_first=True))

        Profilable.__init__(self, "compiler", ["time_profiler", "system_monitor"], specs=self.spec.profilers)

        self.specs = self.find_specifications_by_type("compilation")
        
        n_parallel = self.config.get("n_parallel", 1)
        
        if n_parallel == None:
            self.n_parallel = psutil.cpu_count(logical=True)
        elif n_parallel == "cores":
            self.n_parallel = psutil.cpu_count(logical=False)
        elif n_parallel == "threads":
            self.n_parallel = psutil.cpu_count(logical=True)
        else:
            self.n_parallel = n_parallel

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
        return compile_routine

    def callback(self, retval):
        status, _, res = retval
        if status not in [StatusKind.COMPLETE, None]:
            error_no = MeasureErrorNo.BUILD_TIMEOUT if status == StatusKind.TIMEOUT else MeasureErrorNo.COMPILE_HOST
        else:
            error_no = MeasureErrorNo.NO_ERROR

        if res is not None:
            _, error_msg = res
            if error_msg is not None:
                error_no = MeasureErrorNo.UNKNOWN_ERROR

        self.profiling_checkpoint("run:stop", ctx={"error_no": self.error_map[error_no]})

    def run(self, inputs, idx):
        Experiment.current.set_config({
            "timeout": self.config.get("timeout", 20),
            "n_parallel": self.n_parallel
        })

        stat = "Compiling Tensor Programs: ["
        inp = self.prepare_inputs(inputs)
        pairs = []
        for s in self.specs:
            pairs += s.pair_inputs_options(inp)
        self.profiling_checkpoint("begin")

        results = []
        statuses = []
        workers = [ExecutingWorker() for i in range(self.n_parallel)]
        left_to_do = len(pairs)
        ep = 0
        glob_idx = 0

        while left_to_do > 0:
            num_items = self.n_parallel if self.n_parallel <= left_to_do else left_to_do
            part_pairs = pairs[ep * num_items:(ep + 1) * num_items]
            worker_submissions = []
            part_results = [None] * num_items
            part_statuses = [None] * num_items
            for i, bld in enumerate(part_pairs):
                tp, opts = bld
                tp._load()

                Experiment.current.add_tp_and_opt(
                    {
                        "idx": glob_idx, 
                        "identifier": tp.identifier, 
                        "args": str(tp.args), 
                        "func_name": tp.func_name, 
                        "is_topi": tp.is_topi,
                    },
                    {
                        "idx": glob_idx, 
                        "target": str(opts.target), 
                        "target_host": str(opts.target_host), 
                        "target_options": str(opts.target_opts), 
                        "saving_tensor_program": opts.save_tensor_program, 
                        "saving_source": opts.save_source, 
                        "external_compiler": opts.ext_compiler, 
                        "external_compiler_options": opts.ext_compiler_options,
                        "instruction_count": opts.instr_count, 
                        "pass_configurations": str(opts.pass_configurations), 
                        "implementations_type": str(opts.implementations_type),
                        "implementations_path": str(opts.implementations_path)
                    }
                )

                Experiment.current.set_index(glob_idx)

                self.profiling_checkpoint("start")

                w = workers[i]
                w.submit(
                    self.get_routine(),
                    bld,
                    timeout=self.timeout,
                    callback=self.callback
                )
                worker_submissions.append((i, w))
                glob_idx += 1

            for i, worker in worker_submissions:
                status, _, res = worker.get()
                if status != StatusKind.COMPLETE:
                    result = None
                    if status == StatusKind.TIMEOUT:
                        compile_status = "T"
                    else:
                        compile_status = "E"
                else:
                    _res, error_msg = res
                    if error_msg is not None:
                        compile_status = "E"
                    else:
                        compile_status = "*"
                    result = _res

                part_results[i] = result
                part_statuses[i] = compile_status

            results.extend(part_results)
            statuses.extend(part_statuses)

            left_to_do -= num_items
            ep += 1

        ExecutingWorker.kill_all(workers)
        self.profiling_checkpoint("end")

        stat += "".join(statuses)
        stat += "]"
        logger.info(stat)
        return self.prepapre_outputs(results)