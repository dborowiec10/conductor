from conductor._base import Configurable
from conductor.profiler._base import Profilable
from conductor.experiment import Experiment
from conductor.component.rpc.doppler.client.client import Client
from conductor.component.scheduler.scheduler import schedulers
from conductor.component.evaluator.evaluator import evaluators

import json
import rpyc
import time
import logging
import shutil
import threading

logger = logging.getLogger("conductor.component.runner.remote_parallel")

class RemoteDopplerRunner(Configurable, Profilable):
    _name = "remote_parallel"
    
    def __repr__(self):
        return RemoteDopplerRunner._name

    def __init__(self, evaluator, configs=None, profilers_specs=[]):
        Configurable.__init__(self, "runner", ["remote_doppler"], configs, Configurable.merge_configs({
            "cooldown_interval": 0.1,
            "timeout": 3,
            "n_parallel": 1,
            "rpc_host": "0.0.0.0",
            "rpc_port": 9000,
            "client_host": "0.0.0.0",
            "client_port": 10000
        }, {}, override_first=True))
        Profilable.__init__(self, "runner", ["time_profiler"], specs=profilers_specs)
        self.runner = None
        self._orch_scheduler = None
        self._evaluator = evaluator
        self._profilers_specs = profilers_specs
        
    
    def load(self, num_devs, devices=None):
        # init client side of the runner
        self.c = Client(self.config["client_host"], self.config["client_port"], self.config["rpc_host"], self.config["rpc_port"], devices) 
        # at this point, setters should work
        self.evaluator = self._evaluator
        self.run_type = "regular"
        self.timeout = self.config["timeout"]
        self.cooldown_interval = self.config["cooldown_interval"]
        self.n_parallel = self.config["n_parallel"]
        self.remove_built_schedule = True
        self.dev_ctx_details = self.c
        
        # init server side of the runner (potentially multiple servers via tracker)
        self.c.call_all("r_init_experiment", Experiment.current.store_config, Experiment.current.exp["id"])
        
        pspecs = [{
            "type": p._type, 
            "scope": p.scope, 
            "configuration": p.configuration.to_dict(), 
            "collection": p.collection} for p in self._profilers_specs]
        
        self.c.call_all("r_init_profilers", json.dumps(pspecs))
        
        self.profiling_checkpoint("load")
        
        _ = self.c.call_all("r_load")
        
    def unload(self):
        _ = self.c.call_all("r_unload")
        self.profiling_checkpoint("unload")
        self.c.free()
    
    def load_workers(self, num_devs):
        _ = self.c.call_all("r_load_workers", num_devs)
        
    def load_one_per_dev(self, num_devs=None):
        _ = self.c.call_all("r_load_one_per_dev")

    @property
    def timeout(self):
        return self._timeout
    
    @timeout.setter
    def timeout(self, timeout):
        self._timeout = timeout
        _ = self.c.call_all("r_set_timeout", self._timeout)
        
    @property
    def cooldown_interval(self):
        return self._cooldown_interval
    
    @cooldown_interval.setter
    def cooldown_interval(self, cooldown_interval):
        self._cooldown_interval = cooldown_interval
        _ = self.c.call_all("r_set_cooldown_interval", self._cooldown_interval)
        
    @property
    def remove_built_schedule(self):
        return self._remove_built_schedule
    
    @remove_built_schedule.setter
    def remove_built_schedule(self, remove_built_schedule):
        self._remove_built_schedule = remove_built_schedule
        _ = self.c.call_all("r_set_remove_built_schedule", self._remove_built_schedule)
    
    @property
    def run_type(self):
        return self._run_type
    
    @run_type.setter
    def run_type(self, run_type):
        self._run_type = run_type
        _ = self.c.call_all("r_set_run_type", self._run_type)
        
    @property
    def n_parallel(self):
        return self._n_parallel
    
    @n_parallel.setter
    def n_parallel(self, n_parallel):
        self._n_parallel = n_parallel
        _ = self.c.call_all("r_set_n_parallel", self._n_parallel)
    
    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, target):
        self._target = target
        _ = self.c.call_all("r_set_target", self._target)
    
    def set_target(self, target):
        self.target = target
        
    @property
    def target_host(self):
        return self._target_host
    
    @target_host.setter
    def target_host(self, target_host):
        self._target_host = target_host
        _ = self.c.call_all("r_set_target_host", self._target_host)
        
    def set_target_host(self, target_host):
        self.target_host = target_host

    @property
    def orch_scheduler(self):
        return self._orch_scheduler
    
    @orch_scheduler.setter
    def orch_scheduler(self, orch_scheduler):
        self._orch_scheduler = orch_scheduler
        st = False
        for ks, kv in schedulers.items():
            if isinstance(self._orch_scheduler, kv):
                _ = self.c.call_all("r_set_orch_scheduler", ks)
                st = True
                break
        if not st:
            raise Exception("orch_scheduler not supported")

    def set_orch_scheduler(self, orch_scheduler):
        self.orch_scheduler = orch_scheduler       

    @property
    def evaluator(self):
        return self._evaluator
    
    @evaluator.setter
    def evaluator(self, evaluator):
        self._evaluator = evaluator
        st = False
        for ks, kv in evaluators.items():
            if isinstance(self._evaluator, kv):
                _ = self.c.call_all("r_set_evaluator", ks, json.dumps(self._evaluator.config))
                st = True
                break
        if not st:
            raise Exception("evaluator not supported")

    def profiling_checkpoint(self, checkpoint, ctx=None):
        if checkpoint == "load":
            self.c.call_all("r_profiling_checkpoint", "load", ctx)
        elif checkpoint == "unload":
            self.c.call_all("r_profiling_checkpoint", "unload", ctx)
        elif checkpoint == "run:batch:start":
            self.c.call_all("r_profiling_checkpoint", "run:batch:start", ctx)
        elif checkpoint == "run:batch:stop":
            self.c.call_all("r_profiling_checkpoint", "run:batch:stop", ctx)
        else:
            pass

    def do_run(self, splits, sesskey, results, statuses):
        res, stat = splits["session"].c.root.r_run(
            sesskey,
            splits["configs"],
            splits["build_results"],
            splits["device_ids"],
            splits["files"]
        )
        for r in res:
            results[sesskey].append(self.orch_scheduler.get_measure_result(
                [float(rrr) for rrr in list(r[0])], int(r[1]), r[2], float(r[3]), 
                float(r[4]), float(r[5]), float(r[6]), float(r[7]), r[8]
            ))
            statuses[sesskey].append(r[8])
        

    def run(self, configs, build_results, device_ids, decide_fn=None):
        Experiment.current.set_experiment_stage("runner:run")
        
        self.c.call_all("r_exp_set_task", Experiment.current.current_task["idx"])
        self.c.call_all("r_exp_set_method", Experiment.current.current_method["idx"])
        self.c.call_all("r_exp_set_experiment_stage", Experiment.current.current_stage)
        self.c.call_all("r_sync_experiment")
        self.profiling_checkpoint("run:batch:start")
        
        results = {}
        statuses = {}
        
        stime = time.time()
        stats = "Running candidates: ["
        splits = {}
        for cnf, bld, dev in zip(configs, build_results, device_ids):
            sess = self.c.session_by_device(dev)
            if sess.key not in splits:
                splits[sess.key] = {
                    "session": sess, 
                    "configs": [], 
                    "build_results": [], 
                    "device_ids": [],
                    "files": [],
                }
            if sess.key not in results:
                results[sess.key] = []
            if sess.key not in statuses:
                statuses[sess.key] = []
                
            devspl = dev.split(".")
            
            splits[sess.key]["configs"].append(cnf)
            splits[sess.key]["build_results"].append(bld)
            with open(bld.filename, "rb") as f:
                splits[sess.key]["files"].append(f.read())
            splits[sess.key]["device_ids"].append(devspl[1] + "." + devspl[2])
        
        threads = []
        
        for sesskey in splits.keys():
            t = threading.Thread(target=self.do_run, args=(splits[sesskey], sesskey, results, statuses))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
            
            
        # outs = {}
        results_all = []
        statuses_all = []
        
        for r in results.values():
            results_all += r
        for s in statuses.values():
            statuses_all += s
        
        # for ks, vs in splits.items():
        #     method = rpyc.async_(vs["session"].c.root.r_run)
        #     outs[ks] = method(ks, vs["configs"], vs["build_results"], vs["device_ids"], vs["files"])
        # for ks, vs in splits.items():
        #     outs[ks].wait()
        #     if outs[ks].ready:
        #         out, stat = outs[ks].value
        #         for r in out:
        #             outresults.append(self.orch_scheduler.get_measure_result(
        #                 [float(rrr) for rrr in list(r[0])], int(r[1]), r[2], float(r[3]), 
        #                 float(r[4]), float(r[5]), float(r[6]), float(r[7]), r[8]
        #             ))
        #             statuses.append(r[8])
        stats += "".join(statuses_all)
        stats += "]"
        logger.info(stats)
        run_duration = time.time() - stime
        Experiment.current.add_run_duration(run_duration)
        self.profiling_checkpoint("run:batch:stop")
        return results_all
