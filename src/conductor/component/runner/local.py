from conductor.component.runner._base import Runner
from conductor.experiment import Experiment
from conductor.mediation import MeasureErrorNo
from conductor.component.runner.run_routines.local_run_routine import run_routine
from conductor.component.runner.util import prep_work, select_sub_batch, handle_build_error, handle_result
from conductor._base import Configurable

import time
import logging
import nvtx

logger = logging.getLogger("conductor.component.runner.local")

class LocalRunner(Runner):
    _name = "local"

    def __repr__(self):
        return Runner.__repr__(self) + ":" + LocalRunner._name

    def __init__(self, evaluator, configs=None, profilers_specs=[]):
        Runner.__init__(self, ["local"], evaluator, configs=configs, child_default_configs=Configurable.merge_configs({
            "cooldown_interval": 0.1,
            "timeout": 3,
            "n_parallel": 1
        }, {}, override_first=True), profilers_specs=profilers_specs)
        self.asr = None
        self.dev_ctx_details = None

    def load(self, num_devs, devices=None):
        Runner.load(self, num_devs)

    def unload(self):
        return Runner.unload(self)

    def get_context(self, config, dev_idx):
        run_ctx = Runner.get_ctx(self, dev_idx).copy()
        run_ctx["n_parallel"] = self.n_parallel
        run_ctx["device_key"] = dev_idx
        eval_ctx = self.evaluator.get_ctx()
        return {
            "candidate": config,
            "runner_config": run_ctx,
            "evaluator_config": eval_ctx
        }

    def get_run_routine(self):
        return run_routine

    def run(self, configs, build_results, device_ids, decide_fn=None):
        rng = nvtx.start_range("measure_candidates", color="purple")
        Experiment.current.set_experiment_stage("runner:run")
        self.profiling_checkpoint("run:batch:start")
        stime = time.time()
        stat = "Running candidates: ["
        results = [None] * len(configs)
        statuses = [None] * len(configs)
        
        # prepare work for each device
        left_to_do, unique_dev_ids, work_by_dev = prep_work(configs, build_results, device_ids)
        
        while left_to_do > 0:
            part_work = []
            part_work, ltd, udi = select_sub_batch(left_to_do, unique_dev_ids, work_by_dev, n_parallel=self.n_parallel)
            left_to_do = ltd
            unique_dev_ids = udi
            worker_submissions = []
            # submit batch of work to workers
            for k, (glid, conf, bld, dev_id) in enumerate(part_work):
                self.profiling_checkpoint("run:start", ctx=self.get_context(conf["config_repr"], dev_id))
                if bld.error_no != MeasureErrorNo.NO_ERROR:
                    r, s = handle_build_error(bld, self.callback)
                    results[glid] = self.orch_scheduler.get_measure_result(list(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
                    statuses[glid] = s
                else:
                    w = self.submit_to_worker(bld, conf, dev_id, glid, k)
                    worker_submissions.append((glid, w, bld.time_cost, dev_id))
                
            for glid, worker, btc, dev_id in worker_submissions:
                r, s = handle_result(worker, btc, self.timeout)
                results[glid] = self.orch_scheduler.get_measure_result(list(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
                statuses[glid] = s
                
        stat += "".join(statuses)
        stat += "]"
        logger.info(stat)
        run_duration = time.time() - stime
        Experiment.current.add_run_duration(run_duration)
        self.profiling_checkpoint("run:batch:stop")
        nvtx.end_range(rng)
        return results