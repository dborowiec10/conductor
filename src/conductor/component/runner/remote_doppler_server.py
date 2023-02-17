from conductor.mediation import ERROR_TYPES, MeasureErrorNo
from conductor.worker.worker import ExecutingWorker, StatusKind
from conductor.component.runner.run_routines.local_run_routine import run_routine
from conductor.component.runner.util import prep_work, select_sub_batch, handle_build_error, handle_result
from conductor.component.scheduler.scheduler import schedulers as cond_schedulers
from conductor.component.evaluator.evaluator import evaluators as cond_evaluators
from conductor.profiler._base import Profilable, ProfilerSpecification
from conductor.experiment import Experiment, OrchestratorExperiment
from conductor._base import Configuration
import logging
import hashlib
import json
import os

logger = logging.getLogger("conductor.component.runner.remote_doppler_runner_server")

class RemoteDopplerRunnerServer(Profilable):
    def __init__(self, devices):
        self.orch_schedulers = {}
        self.evaluators = {}
        self.orch_scheduler = None
        self.runner = None
        self.run_type = "regular"  
        self.evaluator = None
        self.target = None
        self.target_host = None
        self.timeout = None
        self.cooldown_interval = None
        self.remove_built_schedule = True
        self.n_parallel = None
        self.worker_process_name = "conductor.worker.worker_process"
        self.error_map = {k:i for k, i in enumerate(ERROR_TYPES)}
        self.devices = list(devices)
        self.workers = []
        self.exp = None

    def exp_init_experiment(self, store_config, exp_id):
        self.exp = OrchestratorExperiment.from_sync(store_config, exp_id)
        Experiment.current = self.exp
    
    def open_file(self, filename, mode):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fh = open(filename, mode)
        return fh
    
    def exp_sync_experiment(self):
        self.exp.sync()
    
    def exp_set_task(self, task_idx):
        self.exp.set_task(task_idx)
        
    def exp_set_method(self, method_idx):
        self.exp.set_method(method_idx)
        
    def exp_set_experiment_stage(self, stage):
        self.exp.set_experiment_stage(stage)
    
    def exp_init_profilers(self, profilers_specs):
        specs = json.loads(profilers_specs)
        pspects = [ProfilerSpecification(s, None, override_config=Configuration(s["configuration"])) for s in specs]
        Profilable.__init__(self, "runner", ["time_profiler"], specs=pspects)

    def set_run_type(self, run_type):
        self.run_type = run_type
    
    def set_n_parallel(self, n_parallel):
        self.n_parallel = n_parallel
    
    def get_run_routine(self):
        return run_routine
    
    def set_timeout(self, timeout):
        self.timeout = timeout
        
    def set_remove_built_schedule(self, remove_built_schedule):
        self.remove_built_schedule = remove_built_schedule
        
    def set_cooldown_interval(self, cooldown_interval):
        self.cooldown_interval = cooldown_interval
    
    def set_target(self, target):
        self.target = target

    def set_target_host(self, target_host):
        self.target_host = target_host

    def set_orch_scheduler(self, orch_scheduler_type):
        if orch_scheduler_type not in self.orch_schedulers:
            self.orch_schedulers[orch_scheduler_type] = cond_schedulers[orch_scheduler_type]()
        self.orch_scheduler = self.orch_schedulers[orch_scheduler_type]

    def set_evaluator(self, evaluator_type, evaluator_config):
        conf = json.loads(evaluator_config)
        hexconf = hashlib.md5(evaluator_config.encode("utf-8")).hexdigest()
        evalkey = evaluator_type + "." + hexconf
        # create evaluator here if needed
        if evalkey not in self.evaluators:
            self.evaluators[evalkey] = cond_evaluators[evaluator_type](configs=conf)
        self.evaluator = self.evaluators[evalkey]

    def load_workers(self, num_devs):
        self.workers = [ExecutingWorker(start=True) for i in range(self.n_parallel * (len(self.devices) if num_devs > len(self.devices) else num_devs))]
    
    def load_one_per_dev(self):
        if len(self.workers) > 0:
            ExecutingWorker.kill_all(self.workers)
        self.workers = [ExecutingWorker(start=True) for i in range(len(self.devices))]
            
    def load(self):
        self.load_workers(len(self.devices))

    def unload(self):
        ExecutingWorker.kill_all(self.workers)
      
    def get_context(self, config, dev_idx):
        return {
            "candidate": config,
            "runner_config": {
                "run_type": self.run_type,
                "timeout": self.timeout,
                "cooldown_interval": self.cooldown_interval,
                "worker_process_name": self.worker_process_name,
                "remove_built_schedule": self.remove_built_schedule,
                "device_key": dev_idx,
                "n_parallel": self.n_parallel
            },
            "evaluator_config": self.evaluator.get_ctx()
        }
    
    def rprofiling_checkpoint(self, checkpoint, ctx):
        self.profiling_checkpoint(checkpoint, ctx=ctx)
        
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
        
    def submit_to_worker(self, bld, conf, dev_id, glid, k):
        # get worker
        if k >= len(self.workers):
            self.workers.append(ExecutingWorker())
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
                None,
                glid,
                self.remove_built_schedule
            ),
            timeout=self.timeout,
            callback=self.callback,
        )
        return w
    
    def run(self, configs, build_results, device_ids, files):
        for f, b in zip(files, build_results):
            dirname = os.path.dirname(b.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(b.filename, "wb") as fp:
                fp.write(f)
                
        results = [None] * len(configs)
        
        # prepare work for each device
        left_to_do, unique_dev_ids, work_by_dev = prep_work(configs, build_results, device_ids)
        
        while left_to_do > 0:
        # select work batch
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
                    results[glid] = r
                else:
                    w = self.submit_to_worker(bld, conf, dev_id, glid, k)
                    worker_submissions.append((glid, w, bld.time_cost, dev_id))

            for glid, worker, btc, dev_id in worker_submissions:
                r, _ = handle_result(worker, btc, self.timeout)
                results[glid] = r
                
        return results