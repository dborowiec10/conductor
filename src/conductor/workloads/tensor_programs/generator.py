import logging
import os

from tvm.autotvm.task.space import ConfigSpace

from conductor.tensor_program import TensorProgram
from conductor.mediation import generic_topi_unwrapper
from conductor.workloads.tensor_programs.testing import tensor_programs as tensor_program_specifications
from conductor.workloads.tensor_programs.definitions import tensor_programs as defined_tensor_programs
from conductor.utils import dict_hash

logger = logging.getLogger("conductor.workloads.tensor_programs.generator")

_target_host = "llvm"
_target_bundles = [("cuda", "cuda"), ("llvm", "x86")]

def generate_tensor_programs(tensor_programs=None, save=False, custom_path=None, target_host=None, target_bundles=None):
    if target_host is not None:
        tgt_host = target_host
    else:
        tgt_host = _target_host
    if target_bundles is not None:
        targets = target_bundles
    else:
        targets = _target_bundles

    tps = tensor_programs if tensor_programs is not None else tensor_program_specifications

    ret_list = []

    for (t, p) in targets:
        for tp_def in tps:
            if len(tp_def) > 2:
                name, inputs, platform_override = tp_def
            else:
                name, inputs = tp_def
                platform_override = None
            
            if platform_override and platform_override != p:
                continue
                
            if len(inputs) == 2:
                inputs_proper, _ = inputs
            else:
                inputs_proper = inputs

            if name in defined_tensor_programs:
                if p in defined_tensor_programs[name]:
                    arg_gen = defined_tensor_programs[name][p]["args"]
                    compute_gen_func = defined_tensor_programs[name][p]["compute_gen_func"]
                    schedule_gen_func = defined_tensor_programs[name][p]["schedule_gen_func"]
                    args = arg_gen(*inputs_proper)
                    identifier = name + "_" + dict_hash(inputs_proper)[:8] + "." + p
                    unwrapped = generic_topi_unwrapper(compute_gen_func, t, identifier)
                    io_tensors = unwrapped(*args)
                    
                    tprog = TensorProgram(
                        identifier, identifier, 
                        compute_gen_func,
                        schedule_gen_func, 
                        args, t, tgt_host,
                        custom_path=os.path.join(custom_path, identifier + ".cond") if custom_path is not None else None,
                        is_topi=True,
                        io_tensors=io_tensors
                    )
                    ret_list.append(tprog)
                    if save:
                        tprog.save(force=True)
    return ret_list
    
                


            



