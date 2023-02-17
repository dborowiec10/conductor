from conductor.component.runner.run_routines.remote_run_routine import run_routine
from conductor.component.runner._base import Runner
from conductor.component.runner.util import prep_work, select_sub_batch, handle_build_error, handle_result
from conductor.experiment import Experiment
from conductor.mediation import MeasureErrorNo
from conductor._base import Configurable
from conductor.component.rpc.default.utils import find_tracker_port
import time
import logging
# import nvtx

logger = logging.getLogger("conductor.component.runner.remote_tvm")

class RemoteTVMRunner(Runner):
    _name = "remote_tvm"

    def __repr__(self):
        return Runner.__repr__(self) + ":" + RemoteTVMRunner._name

    def __init__(self, evaluator, configs=None, profilers_specs=[]):
        Runner.__init__(self, ["remote_tvm"], evaluator, configs=configs, child_default_configs=Configurable.merge_configs({
            "cooldown_interval": 0.1,
            "timeout": 3,
            "n_parallel": 1,
            "rpc_host": "0.0.0.0",
            "rpc_port": 9000,
            "rpc_port_end": 9199
        }, {}, override_first=True), profilers_specs=profilers_specs)
        self.asr = None
        self.tracker_port = find_tracker_port(self.config["rpc_host"], self.config["rpc_port"], self.config["rpc_port_end"])
        print("FIND TRACKER PORT", self.tracker_port)
        self.dev_ctx_details = None
        

    def load(self, num_devs, devices=None):
        print("Runner Load Devices", devices)
        unique_key = ""
        for d in devices:
            spl = d.split(".")
            if unique_key == "":
                unique_key = spl[0]
            else:
                if unique_key != spl[0]:
                    raise Exception("All devices must be on the same key")
                else:
                    unique_key = spl[0]
                    
        self.dev_ctx_details = (unique_key, self.config["rpc_host"], self.tracker_port)
        Runner.load(self, num_devs)

    def unload(self):
        return Runner.unload(self)

    def get_context(self, config, dev_idx):
        run_ctx = Runner.get_ctx(self, dev_idx).copy()
        run_ctx["n_parallel"] = self.n_parallel
        run_ctx["rpc_host"] = self.config["rpc_host"]
        run_ctx["rpc_port"] = self.config["rpc_port"]
        run_ctx["rpc_port_end"] = self.config["rpc_port_end"]
        run_ctx["rpc_tracker_port"] = self.tracker_port
        eval_ctx = self.evaluator.get_ctx()
        return {
            "candidate": config,
            "runner_config": run_ctx,
            "evaluator_config": eval_ctx
        }

    def get_run_routine(self):
        return run_routine
    
    def run(self, configs, build_results, device_ids, decide_fn=None):
        # rng = nvtx.start_range("measure_candidates", color="purple")
        Experiment.current.set_experiment_stage("runner:run")
        self.profiling_checkpoint("run:batch:start")
        stime = time.time()
        stat = "Running candidates: ["
        results = [None] * len(configs)
        statuses = [None] * len(configs)
        
        left_to_do, unique_dev_ids, work_by_dev = prep_work(configs, build_results, device_ids)
        
        while left_to_do > 0:
            part_work = []
            part_work, ltd, udi = select_sub_batch(left_to_do, unique_dev_ids, work_by_dev, n_parallel=self.n_parallel)
            left_to_do = ltd
            unique_dev_ids = udi
            worker_submissions = []
            for k, (glid, conf, bld, dev_id) in enumerate(part_work):

                self.profiling_checkpoint("run:start", ctx=self.get_context(conf["config_repr"], dev_id))
                if bld.error_no != MeasureErrorNo.NO_ERROR:
                    r, s = handle_build_error(bld, self.callback)
                    results[glid] = self.orch_scheduler.get_measure_result(list(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
                    statuses[glid] = s
                else:
                    # we set dev_idx to 0 when submitting to ensure that sess.device("cuda", dev_id) is correct and points to 0 (considering TVM's RPC infra) - e.g. each server allocated to 1 device
                    # this is necessary because of how CUDA_VISIBLE_DEVICES enumerates devices. For example, if CUDA_VISIBLE_DEVICES is set to 1,2,3, the TVM's RPC server will see this as dev("cuda", 0), dev("cuda", 1), dev("cuda", 2)
                    w = self.submit_to_worker(bld, conf, "stub.0", glid, k, ctx=self.dev_ctx_details)
                    worker_submissions.append((glid, w, bld.time_cost))
                
            for glid, worker, btc in worker_submissions:
                r, s = handle_result(worker, btc, self.timeout)
                results[glid] = self.orch_scheduler.get_measure_result(list(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
                statuses[glid] = s
                
        stat += "".join(statuses)
        stat += "]"
        logger.info(stat)
        run_duration = time.time() - stime
        Experiment.current.add_run_duration(run_duration)
        self.profiling_checkpoint("run:batch:stop")
        # nvtx.end_range(rng)
        return results