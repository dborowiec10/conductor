from conductor.utils import request_remote
from conductor.component.scheduler.encode import look_call
from conductor.mediation import BuildResult, MeasureResult, MeasureInput, MeasureErrorNo, Tasker, ERROR_TYPES

import tvm
import numpy as np
import hashlib

import logging

logger = logging.getLogger("conductor.component.scheduler")

class Scheduler(object):
    _name = "scheduler"

    def __repr__(self):
        return Scheduler._name

    def get_build_result(self, filename, args, error_no, error_msg, time_cost, timestamp, status):
        return BuildResult(filename, args, error_no, error_msg, time_cost, timestamp, status)

    def get_measure_result(self, costs, error_no, error_msg, time_cost, timestamp, achieved_flop, mean, total_time, status):
        return MeasureResult(costs, error_no, error_msg, time_cost, timestamp, achieved_flop, mean, total_time, status)

    def measure_input_to_config(self, inp):
        return inp.config

    def get_mean_cost(self, res):
        if isinstance(res, MeasureResult):
            return res.mean
        else:
            logger.error("unable to retrieve mean for res: %s", str(res))
            return 1e20

    def get_build_inputs(self, task, configs, device_ids, dev_ctx_details, hash_callback, options=None):
        theoretical_flop = Tasker.task_theoretical_flop(task)
        unique_dev_ids = list(set(device_ids))
        sm_map = {k: self.get_sm(dev_ctx_details, task.target, k) for k in unique_dev_ids}
        build_inputs = []
        temp_configs = []
        for k, (c, dev_id) in enumerate(zip(configs, device_ids)):
            temp_config = {"config": c, "config_repr": str(c), "flop": theoretical_flop}
            temp_configs.append(temp_config)
            build_inputs.append({
                "orch_scheduler": self,
                "task": task,
                "config": temp_config,
                "options": {
                    "options": options,
                    "idx": k
                },
                "sm": sm_map[dev_id],
                "verify": self.get_build_options(dev_ctx_details, task.target, dev_id),
                "hash_callback": hash_callback
            })
        return build_inputs, temp_configs, theoretical_flop

    def get_measure_input(self, bld_res, theoretical_flop, target, device_id, cooldown_interval, evaluator, dev_ctx_details, glob_idx, remove_built_schedule):
        return {
            "_build_res": self.get_build_result(*bld_res),
            "flop": theoretical_flop,
            "target": target,
            "device_id": device_id,
            "cooldown_interval": cooldown_interval,
            "evaluator": evaluator,
            "dev_ctx_details": dev_ctx_details,
            "idx": glob_idx,
            "remove": remove_built_schedule
        }

    def get_inp_res_err(self, configs, results, theoretical_flop, task):
        tot_error_count = 0
        error_counts = [0 for i in range(len(ERROR_TYPES))]
        m_inps = []
        m_res = []
        for k, (conf, res) in enumerate(zip(configs, results)):
            if res.error_no != MeasureErrorNo.NO_ERROR:
                tot_error_count += 1
            error_counts[res.error_no] += 1
            m_inps.append(MeasureInput(task, Tasker.task_to_key(task), task.target, conf, theoretical_flop))
            m_res.append(res)
        return m_inps, m_res, m_inps, m_res, tot_error_count, error_counts

    # generate schedule given a schedulable object
    def from_schedulable(self):
        raise NotImplementedError()

    # generate schedule given task and config
    def from_task(self):
        raise NotImplementedError()

    def get_tvm_arguments(self, args, ctx):
        tvm_arguments = []
        for a in args:
            if isinstance(a, tvm.te.tensor.Tensor):
                tens_shape = tuple([int(x) for x in list(a.shape)])
                if a.op.name == "placeholder":
                    np_arg = np.random.uniform(size=tens_shape).astype(a.dtype)
                    tvm_arg = tvm.nd.array(np_arg, ctx=ctx)
                    tvm_arguments.append(tvm_arg)
                else:
                    tvm_arg = tvm.nd.empty(tens_shape, dtype=a.dtype, ctx=ctx)
                    tvm_arguments.append(tvm_arg)

        return tvm_arguments

    def sm_from_ctx(self, ctx, tgt):
        if ctx.exist:
            if ("cuda" in tgt.keys or "opencl" in tgt.keys or "rocm" in tgt.keys or "vulkan" in tgt.keys):
                sm = ("sm_" + "".join(ctx.compute_version.split(".")))
            else:
                sm = None
        else:
            raise RuntimeError("Could not find a context!")
        return sm

    def build_options_from_ctx(self, ctx, tgt):
        options = {}
        if ctx.exist:
            if ("cuda" in tgt.keys or "opencl" in tgt.keys or "rocm" in tgt.keys or "vulkan" in tgt.keys):
                max_dims = ctx.max_thread_dimensions
                options["check_gpu"] = {
                    "max_shared_memory_per_block": ctx.max_shared_memory_per_block,
                    "max_threads_per_block": ctx.max_threads_per_block,
                    "max_thread_x": max_dims[0],
                    "max_thread_y": max_dims[1],
                    "max_thread_z": max_dims[2],
                }
        else:
            raise RuntimeError("Could not find a context!")
        return options

    def get_sm(self, dev_ctx_details, target, device_id):
        if isinstance(target, str):
            tgt = tvm.target.Target(target)
        else:
            tgt = target
        # extract numerical device id from key
        devid = int(device_id.split(".")[-1])
        if dev_ctx_details == None:
            # local device
            if "cuda" in tgt.keys:
                ctx = tvm.cuda(devid)
            else:
                ctx = tvm.cpu(devid)
            sm = self.sm_from_ctx(ctx, tgt)
        else:
            if isinstance(dev_ctx_details, tuple):
                # default (TVM) rpc
                newctx = dev_ctx_details
                print("NEWCTX", newctx)
                remote = request_remote(*newctx)
                # WARNING: 0 is a hack - see comments about CUDA_VISIBLE_DEVICES and enumeration with TVM RPC
                ctx = remote.device("cuda" if "cuda" in tgt.keys else "llvm", 0)
                sm = self.sm_from_ctx(ctx, tgt)
            else:
                # doppler rpc
                sm = dev_ctx_details.get_sm(device_id, str(tgt))
        return sm

    def get_build_options(self, dev_ctx_details, target, device_id):
        if isinstance(target, str):
            tgt = tvm.target.Target(target)
        else:
            tgt = target
        # extract numerical device id from key
        devid = int(device_id.split(".")[-1])
        if dev_ctx_details == None:
            if "cuda" in tgt.keys:
                ctx = tvm.cuda(devid)
            else:
                ctx = tvm.cpu(devid)
            options = self.build_options_from_ctx(ctx, tgt)
        else:
            if isinstance(dev_ctx_details, tuple):
                newctx = dev_ctx_details
                remote = request_remote(*newctx)
                # WARNING: 0 is a hack - see comments about CUDA_VISIBLE_DEVICES and enumeration with TVM RPC
                ctx = remote.device("cuda" if "cuda" in tgt.keys else "llvm", 0)
                options = self.build_options_from_ctx(ctx, tgt)
            else:
                options = dev_ctx_details.get_build_options(device_id, str(tgt))
        return options

    def encode_schedule(self, sch):
        storage = {"ops": {}, "tensors": {}}
        s_stgdone = ""
        for k, v in enumerate(sch.stages):
            s_stgdone += look_call(v, storage)
        s_stgdone = s_stgdone. \
            replace("compute", "C"). \
            replace("placeholder", "P"). \
            replace(" ", ""). \
            replace(".", ""). \
            replace("inner", "i"). \
            replace("outer", "o"). \
            replace("fused", "f"). \
            replace("parent", "p"). \
            replace("global", "g"). \
            replace("auto_unroll_max_step", "aums")

        # return s_stgdone
        return hashlib.sha512(s_stgdone.encode("ascii")).hexdigest()