from conductor._base import Configurable
from profiler._base import Profilable
from conductor.mediation import ERROR_TYPES, MeasureErrorNo
from conductor.worker.worker import ExecutingWorker, StatusKind
from conductor.component.scheduler._base import Scheduler

import time
import logging
logger = logging.getLogger("conductor.component.runner")

class Runner(Configurable, Profilable):
    _name = "runner"
    
    def __repr__(self):
        return Runner._name

    def __init__(self, subtypes, evaluator, configs=None, child_default_configs={}, profilers_specs=[]):
        Configurable.__init__(self, "runner", subtypes, configs, Configurable.merge_configs({
            "cooldown_interval": 0.1,
            "timeout": 3,
            "n_parallel": 1
        }, child_default_configs, override_first=True))

        Profilable.__init__(self, "runner", ["time_profiler"], specs=profilers_specs)
        self.orch_scheduler: Scheduler = None
        self.runner = None
        self.run_type = "regular"  
        self.evaluator = evaluator
        self.target = None
        self.target_host = None
        self.timeout = self.config["timeout"]
        self.n_parallel = self.config["n_parallel"]
        self.cooldown_interval = self.config["cooldown_interval"]
        self.remove_built_schedule = True
        self.worker_process_name = "conductor.worker.worker_process"
        self.error_map = {k:i for k, i in enumerate(ERROR_TYPES)}

    def set_target(self, target):
        self.target = target

    def set_target_host(self, target_host):
        self.target_host = target_host

    def set_orch_scheduler(self, orch_scheduler):
        self.orch_scheduler = orch_scheduler

    def get_ctx(self, dev_idx):
        return {
            "run_type": self.run_type,
            "timeout": self.timeout,
            "cooldown_interval": self.cooldown_interval,
            "worker_process_name": self.worker_process_name,
            "remove_built_schedule": self.remove_built_schedule
        }

    def profiling_checkpoint(self, checkpoint, ctx=None):
        if checkpoint == "load":
            self.begin_profilers()
        elif checkpoint == "unload":
            self.end_profilers()
        elif checkpoint == "run:batch:start":
            pass
        elif checkpoint == "run:batch:stop":
            self.persist_profilers()
        elif checkpoint == "run:start":
            self.checkpoint_data["run:start"] = self.start_profilers(context=ctx)
        elif checkpoint == "run:stop":
            self.append_context_profilers(self.checkpoint_data["run:start"], ctx)
            self.stop_profilers(self.checkpoint_data["run:start"])
        else:
            pass

    def load(self, num_devs):
        self.profiling_checkpoint("load")
        self.load_workers(num_devs)

    def load_workers(self, num_devs):
        self.workers = [ExecutingWorker() for i in range(self.n_parallel * num_devs)]

    def load_one_per_dev(self, num_devs=None):
        ExecutingWorker.kill_all(self.workers)
        self.workers = [ExecutingWorker() for i in range(num_devs)]

    def unload(self):
        ExecutingWorker.kill_all(self.workers)
        self.profiling_checkpoint("unload")

    def callback(self, retval):
        status, err_msg, value = retval
        if status not in [StatusKind.COMPLETE, None]:
            error_no = MeasureErrorNo.RUN_TIMEOUT if status == StatusKind.TIMEOUT else MeasureErrorNo.RUNTIME_DEVICE
            mean = 1e20
            total_time = 1e20
            achieved_flop = 0
            other = None
        else:
            _, error_no, _, _, _, mean, total_time, other, achieved_flop = value
        self.profiling_checkpoint("run:stop", ctx={
            "status": self.error_map[error_no],
            "measurement_mean": mean,
            "measurement_total_time": total_time,
            "achieved_flop": achieved_flop,
            "other": other
        })

    def submit_to_worker(self, bld, conf, dev_id, glid, k, ctx=None):
        # get worker
        w = self.workers[k]
        if w.status == False:
            self.workers[k] = ExecutingWorker()
            w = self.workers[k]
        # submit to worker
        w.submit(
            self.get_run_routine(),
            self.orch_scheduler.get_measure_input(
                bld,
                conf["flop"],
                self.target,
                int(dev_id.split(".")[-1]),
                self.cooldown_interval,
                self.evaluator,
                ctx,
                glid,
                self.remove_built_schedule
            ),
            timeout=self.timeout,
            callback=self.callback,
        )
        return w