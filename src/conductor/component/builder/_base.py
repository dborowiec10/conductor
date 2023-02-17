
from conductor._base import Configurable
from conductor.profiler._base import Profilable
from conductor.mediation import MeasureErrorNo, ERROR_TYPES
from conductor.component.builder.build_routine import build_routine
from conductor.worker.worker import ExecutingWorker, StatusKind

import tvm
import logging
logger = logging.getLogger("conductor.orchestrator.builder")

class Builder(Configurable, Profilable):
    _name = "builder"
    
    def __repr__(self):
        return Builder._name

    def __init__(self, subtype, configs=None, child_default_configs={}, profilers_specs=[]):
        Configurable.__init__(self, "builder", [subtype], configs, Configurable.merge_configs({
            "timeout": 10,
            "build_func": "default"
        }, child_default_configs, override_first=True))

        Profilable.__init__(self, "builder", ["time_profiler"], specs=profilers_specs)
        self.timeout = self.config["timeout"]
        self.n_parallel = 1
        self.build_func = self.config["build_func"]
        self.builder = tvm.auto_scheduler.measure.LocalBuilder(
            timeout=self.timeout,
            n_parallel=self.n_parallel,
            build_func="default"
        )
        self.orch_scheduler = None
        self.worker_process_name = "conductor.worker.worker_process"
        self.error_map = {k:i for k, i in enumerate(ERROR_TYPES)}

    def profiling_checkpoint(self, checkpoint, ctx=None):
        if checkpoint == "load":
            self.begin_profilers()
            
        elif checkpoint == "unload":
            self.end_profilers()

        elif checkpoint == "build:batch:start":
            pass

        elif checkpoint == "build:batch:stop":
            self.persist_profilers()

        elif checkpoint == "build:start":
            self.checkpoint_data["build:start"] = self.start_profilers(context=ctx)

        elif checkpoint == "build:stop":
            self.append_context_profilers(self.checkpoint_data["build:start"], ctx)
            self.stop_profilers(self.checkpoint_data["build:start"])
        else:
            pass

    def load(self):
        self.profiling_checkpoint("load")
        self.workers = [ExecutingWorker() for i in range(self.n_parallel)]

    def unload(self):
        ExecutingWorker.kill_all(self.workers)
        self.profiling_checkpoint("unload")

    def get_build_routine(self):
        return build_routine

    def set_orch_scheduler(self, orch_scheduler):
        self.orch_scheduler = orch_scheduler

    def callback(self, retval):
        status, err_msg, value = retval
        if status not in [StatusKind.COMPLETE, None]:
            error_no = MeasureErrorNo.BUILD_TIMEOUT if status == StatusKind.TIMEOUT else MeasureErrorNo.COMPILE_HOST
        else:
            _, _, error_no, _, _, _ = value
        self.profiling_checkpoint("build:stop", ctx={"status": self.error_map[error_no]})


