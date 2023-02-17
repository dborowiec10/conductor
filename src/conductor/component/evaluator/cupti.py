from conductor._base import Configurable
from conductor.component.evaluator._base import Evaluator
from conductor.mediation import ProfileResult

import tvm
import json
import numpy as np

import logging
logger = logging.getLogger("conductor.component.evaluator.cupti")

class CUPTIEvaluator(Evaluator):
    _name = "cupti"

    def __repr__(self):
        return Evaluator.__repr__(self) + ":" + CUPTIEvaluator._name

    def __init__(self, configs=None):
        Evaluator.__init__(self, ["cupti"], configs=configs, child_default_configs=Configurable.merge_configs({
            "num_avg_runs": 10,
            "num_measure_repeat": 1,
            "min_repeat_ms": 0,
            "enable_cpu_cache_flush": False,
            "metrics": [
                {"name": "dram_read_transactions", "enabled": True},
                {"name": "dram_write_transactions", "enabled": True},
                {"name": "dram_read_throughput", "enabled": True},
                {"name": "dram_write_throughput", "enabled": True},
                {"name": "dram_utilization", "enabled": False},
                {"name": "dram_read_bytes", "enabled": False},
                {"name": "dram_write_bytes", "enabled": False},
                {"name": "sysmem_write_transactions", "enabled": False},
                {"name": "sysmem_write_throughput", "enabled": False},
                {"name": "sysmem_write_utilization", "enabled": False},
                {"name": "sysmem_write_bytes", "enabled": False},
                {"name": "sysmem_utilization", "enabled": False},
                {"name": "sysmem_read_throughput", "enabled": False},
                {"name": "sysmem_read_utilization", "enabled": False},
                {"name": "sysmem_read_bytes", "enabled": False},
                {"name": "sysmem_read_transactions", "enabled": False},
                {"name": "flop_hp_efficiency", "enabled": False},
                {"name": "flop_sp_efficiency", "enabled": False},
                {"name": "flop_dp_efficiency", "enabled": False},
                {"name": "flop_count_hp", "enabled": False},
                {"name": "flop_count_hp_add", "enabled": False},
                {"name": "flop_count_hp_mul", "enabled": False},
                {"name": "flop_count_hp_fma", "enabled": False},
                {"name": "flop_count_dp_mul", "enabled": False},
                {"name": "flop_count_dp", "enabled": False},
                {"name": "flop_count_dp_add", "enabled": False},
                {"name": "flop_count_dp_fma", "enabled": False},
                {"name": "flop_count_sp", "enabled": False},
                {"name": "flop_count_sp_add", "enabled": False},
                {"name": "flop_count_sp_fma", "enabled": False},
                {"name": "flop_count_sp_mul", "enabled": False},
                {"name": "flop_count_sp_special", "enabled": False},
                {"name": "global_hit_rate", "enabled": False},
                {"name": "gld_throughput", "enabled": False},
                {"name": "gld_requested_throughput", "enabled": False},
                {"name": "gld_efficiency", "enabled": False},
                {"name": "gld_transactions", "enabled": False},
                {"name": "gld_transactions_per_request", "enabled": False},
                {"name": "gst_throughput", "enabled": False},
                {"name": "gst_requested_throughput", "enabled": False},
                {"name": "gst_efficiency", "enabled": False},
                {"name": "gst_transactions", "enabled": False},
                {"name": "gst_transactions_per_request", "enabled": False},
                {"name": "inst_executed_global_stores", "enabled": False},
                {"name": "inst_executed_global_loads", "enabled": False},
                {"name": "stall_inst_fetch", "enabled": False},
                {"name": "stall_exec_dependency", "enabled": False},
                {"name": "stall_memory_dependency", "enabled": False},
                {"name": "stall_sync", "enabled": False},
                {"name": "stall_other", "enabled": False},
                {"name": "stall_constant_memory_dependency", "enabled": False},
                {"name": "stall_pipe_busy", "enabled": False},
                {"name": "stall_memory_throttle", "enabled": False},
                {"name": "stall_not_selected", "enabled": False},
                {"name": "inst_fp_16", "enabled": False},
                {"name": "inst_fp_32", "enabled": False},
                {"name": "inst_fp_64", "enabled": False},
                {"name": "inst_integer", "enabled": False},
                {"name": "inst_bit_convert", "enabled": False},
                {"name": "inst_control", "enabled": False},
                {"name": "inst_compute_ld_st", "enabled": False},
                {"name": "inst_misc", "enabled": False},
                {"name": "inst_inter_thread_communication", "enabled": False},
                {"name": "inst_executed_global_reductions", "enabled": False},
                {"name": "inst_executed_surface_stores", "enabled": False},
                {"name": "inst_executed_surface_reductions", "enabled": False},
                {"name": "inst_executed_surface_loads", "enabled": False},
                {"name": "warp_execution_efficiency", "enabled": False},
                {"name": "warp_nonpred_execution_efficiency", "enabled": False},
                {"name": "inst_per_warp", "enabled": False},
                {"name": "unique_warps_launched", "enabled": False},
                {"name": "inst_executed", "enabled": False},
                {"name": "inst_replay_overhead", "enabled": False},
                {"name": "inst_issued", "enabled": False},
                {"name": "issue_slots", "enabled": False},
                {"name": "ldst_fu_utilization", "enabled": False},
                {"name": "ldst_issued", "enabled": False},
                {"name": "ldst_executed", "enabled": False},
                {"name": "cf_fu_utilization", "enabled": False},
                {"name": "cf_issued", "enabled": False},
                {"name": "cf_executed", "enabled": False},
                {"name": "single_precision_fu_utilization", "enabled": False},
                {"name": "double_precision_fu_utilization", "enabled": False},
                {"name": "half_precision_fu_utilization", "enabled": False},
                {"name": "inst_executed_shared_loads", "enabled": False},
                {"name": "inst_executed_shared_stores", "enabled": False},
                {"name": "shared_load_transactions_per_request", "enabled": False},
                {"name": "shared_load_transactions", "enabled": False},
                {"name": "shared_load_throughput", "enabled": False},
                {"name": "shared_store_throughput", "enabled": False},
                {"name": "shared_store_transactions", "enabled": False},
                {"name": "shared_store_transactions_per_request", "enabled": False},
                {"name": "l2_read_transactions", "enabled": False},
                {"name": "l2_read_throughput", "enabled": False},
                {"name": "l2_write_transactions", "enabled": False},
                {"name": "l2_write_throughput", "enabled": False},
                {"name": "l2_utilization", "enabled": False},
                {"name": "sm_efficiency", "enabled": False},
                {"name": "achieved_occupancy", "enabled": False},
                {"name": "pcie_total_data_transmitted", "enabled": False},
                {"name": "pcie_total_data_received", "enabled": False}
            ],
            "events": [
                { "name": "active_cycles", "enabled": False },
                { "name": "active_cycles_pm", "enabled": False },
                { "name": "active_warps", "enabled": False },
                { "name": "active_warps_pm", "enabled": False },
                { "name": "elapsed_cycles_pm", "enabled": False },
                { "name": "elapsed_cycles_sm", "enabled": False },
                { "name": "fb_subp0_read_sectors", "enabled": False },
                { "name": "fb_subp1_read_sectors", "enabled": False },
                { "name": "fb_subp0_write_sectors", "enabled": False },
                { "name": "fb_subp1_write_sectors", "enabled": False },
                { "name": "inst_executed", "enabled": False },
                { "name": "inst_issued1", "enabled": False },
                { "name": "l2_subp0_read_sector_misses", "enabled": False },
                { "name": "l2_subp1_read_sector_misses", "enabled": False },
                { "name": "l2_subp0_read_sysmem_sector_queries", "enabled": False },
                { "name": "l2_subp1_read_sysmem_sector_queries", "enabled": False },
                { "name": "l2_subp0_read_tex_sector_queries", "enabled": False },
                { "name": "l2_subp1_read_tex_sector_queries", "enabled": False },
                { "name": "l2_subp0_total_read_sector_queries", "enabled": False },
                { "name": "l2_subp1_total_read_sector_queries", "enabled": False },
                { "name": "l2_subp0_total_write_sector_queries", "enabled": False },
                { "name": "l2_subp1_total_write_sector_queries", "enabled": False },
                { "name": "l2_subp0_write_sector_misses", "enabled": False },
                { "name": "l2_subp1_write_sector_misses", "enabled": False },
                { "name": "l2_subp0_write_sysmem_sector_queries", "enabled": False },
                { "name": "l2_subp1_write_sysmem_sector_queries", "enabled": False },
                { "name": "l2_subp0_write_tex_hit_sectors", "enabled": False },
                { "name": "l2_subp1_write_tex_hit_sectors", "enabled": False },
                { "name": "l2_subp0_write_tex_sector_queries", "enabled": False },
                { "name": "l2_subp1_write_tex_sector_queries", "enabled": False },
                { "name": "not_predicated_off_thread_inst_executed", "enabled": False },
                { "name": "inst_issued0", "enabled": False },
                { "name": "shared_load", "enabled": False },
                { "name": "shared_store", "enabled": False },
                { "name": "generic_load", "enabled": False },
                { "name": "generic_store", "enabled": False },
                { "name": "global_load", "enabled": False },
                { "name": "global_store", "enabled": False },
                { "name": "local_load", "enabled": False },
                { "name": "local_store", "enabled": False },
                { "name": "shared_ld_transactions", "enabled": False },
                { "name": "shared_st_transactions", "enabled": False },
                { "name": "thread_inst_executed", "enabled": False },
                { "name": "warps_launched", "enabled": False }
            ]
        }, {}, override_first=True))

    def get_ctx(self):
        return super().get_ctx()

    def evaluate(self, mod_func, mod_func_name, flop, ctx, args, fname=None):
        fcreate = tvm._ffi.get_global_func("runtime.RPCCUPTIEvaluator")
        feval = fcreate(
            mod_func, mod_func_name, 
            ctx.device_type, ctx.device_id, 
            self.number,
            self.repeat,
            self.min_repeat_ms,
            self.f_preproc,
            json.dumps({
                "metrics": self.config["metrics"],
                "events": self.config["events"]
            })
        )
        def evaluator(*args):
            """Internal wrapped evaluator."""
            blob = feval(*args)
            out = json.loads(blob)
            return ProfileResult(mean=np.mean(out["costs"]), results=out["costs"], total_time=out["total_time"], other=out["other"])
            
        e = evaluator(*args)
        return (e.results, e.mean, e.total_time, e.other)
