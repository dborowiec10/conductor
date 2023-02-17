from conductor.component.builder._base import Builder
from conductor.mediation import MeasureErrorNo
from conductor._base import Configurable
from conductor.worker.worker import ExecutingWorker, StatusKind
from conductor.experiment import Experiment
import time

import logging
logger = logging.getLogger("conductor.component.builder.serial")

class SerialBuilder(Builder):
    _name = "serial"
    
    def __repr__(self):
        return Builder.__repr__(self) + ":" + SerialBuilder._name

    def __init__(self, configs=None, profilers_specs=[]):
        Builder.__init__(self, "serial", configs=configs, child_default_configs=Configurable.merge_configs({
            "timeout": 10,
            "build_func": "default"
        },{}, override_first=True), profilers_specs=profilers_specs)
        self.n_parallel = 1

    def build(self, build_inputs):
        Experiment.current.set_experiment_stage("builder:build")
        self.profiling_checkpoint("build:batch:start")
        stime = time.time()
        stat = "Building candidates: ["
        results = []
        statuses = []
        ep = 0
        left_to_do = len(build_inputs)
        while left_to_do > 0:
            num_items = self.n_parallel if self.n_parallel <= left_to_do else left_to_do
            part_build_inputs = build_inputs[ep * num_items:(ep + 1) * num_items]
            part_results = [None] * len(part_build_inputs)
            part_statuses = [None] * len(part_build_inputs)
            worker_submissions = []

            for idx, bld in enumerate(part_build_inputs):
                self.profiling_checkpoint("build:start", ctx={
                    "builder_config": {
                        "name": "parallel",
                        "n_parallel": self.n_parallel,
                        "timeout": self.timeout,
                        "candidate": bld["config"]["config_repr"],
                        "build_func": self.config["build_func"]
                    }
                })
                w = self.workers[idx]
                if w.status == False:
                    self.workers[idx] = ExecutingWorker()
                    w = self.workers[idx]
                w.submit(
                    self.get_build_routine(),
                    bld,
                    timeout=self.timeout,
                    callback=self.callback,
                )
                worker_submissions.append((idx, w))

            for idx, worker in worker_submissions:
                status, err_msg, res = worker.get()
                if status != StatusKind.COMPLETE:
                    error_msg = err_msg
                    filename = None
                    args = None
                    time_cost = self.timeout
                    timestamp = time.time()
                    if status == StatusKind.TIMEOUT:
                        error_no = MeasureErrorNo.BUILD_TIMEOUT
                        status = "T"
                    else:
                        error_no = MeasureErrorNo.COMPILE_HOST
                        status = "E"
                else:
                    filename, args, error_no, error_msg, time_cost, timestamp = res
                    status = "*" if error_no == MeasureErrorNo.NO_ERROR else "E"

                part_results[idx] = self.orch_scheduler.get_build_result(filename, args, error_no, error_msg, time_cost, timestamp, status)
                part_statuses[idx] = status

            results.extend(part_results)
            statuses.extend(part_statuses)

            left_to_do -= num_items
            ep += 1

        stat += "".join(statuses)
        stat += "]"
        logger.info(stat)
        build_duration = time.time() - stime
        Experiment.current.add_build_duration(build_duration)
        self.profiling_checkpoint("build:batch:stop")
        return results
