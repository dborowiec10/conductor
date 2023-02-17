from generator.generate import get_job
import os
import json
import copy

base = os.path.dirname(os.path.realpath(__file__))
    
workloads_tp = {
    "cuda": [
        {"name": "sh_0", "type": "tensor_program", "identifier": "batch_matmul_51268830.cuda"},
        {"name": "sh_6", "type": "tensor_program", "identifier": "conv1d_ncw_f691e8dc.cuda"},
        {"name": "sh_19", "type": "tensor_program", "identifier": "conv2d_nchw_607f6363.cuda"},
        {"name": "sh_29", "type": "tensor_program", "identifier": "conv3d_ncdhw_59c7a972.cuda"},
        {"name": "sh_34", "type": "tensor_program", "identifier": "correlation_nchw_b6bb56a9.cuda"},
        {"name": "sh_35", "type": "tensor_program", "identifier": "dense_small_batch_21b6da02.cuda"}
    ]
}

autotuner_serial = {
    "ansor": {   
        "orchestrator": "default", 
        "strategy": "gradient",
        "method_type": "composite_sketch",
        "method_name": None,           
        "cost_model": "ansor_xgb",
        "optimizer": None,                     
        "search_policy": "ansor_sketch", 
        "sampler": None,       
        "filter": None,              
        "measurer": "default", 
        "evaluator": "default",  
        "builder": "parallel", 
        "runner": "remote_tvm"
    },
    "grid_index": {
        "orchestrator": "default", 
        "strategy": "sequential", 
        "method_type": "standalone_template", 
        "method_name": "grid_index",
        "cost_model": None,                   
        "optimizer": None,                     
        "search_policy": None,           
        "sampler": None,       
        "filter": None,              
        "measurer": "default", 
        "evaluator": "default",  
        "builder": "parallel", 
        "runner": "remote_tvm"
    },
}

base_save_path = os.path.join(base, "serial_multi_gpu_rpc_local")

def save(dir, job, filename):
    dirpath = os.path.join(base_save_path, dir)
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w") as f:
        json.dump(job, f, indent=4)
        
settings = {
    "num_trials": 500,
    "batch_size": 64,
    "early_stop": None
}

devices = {
    "2": ["localmachine:2080.0", "localmachine:2080.1"],
    "3": ["localmachine:2080.0", "localmachine:2080.1", "localmachine:2080.2"],
    "4": ["localmachine:2080.0", "localmachine:2080.1", "localmachine:2080.2", "localmachine:2080.3"],
}

# assumption is that devices are remote, hence cannot measure the gpu-related profiling stats via nvml
# doppler remote runner tackles this
gpus = {
    "2": [],
    "3": [],
    "4": []
}

platform = "intelv100"
dop = "doppler"
approach = "serial_mutli_gpu_rpc_local"

def generate_multi_gpu_rpc():
    for ((k_gpus, v_gpus), (k_devices, v_devices)) in zip(gpus.items(), devices.items()):
        for wptcuda in workloads_tp["cuda"]:
            for atk, atv in autotuner_serial.items():
                d, j, fname = get_job(
                    int(wptcuda["name"].split("_")[-1]), 
                    wptcuda, 
                    atk, 
                    atv, 
                    settings,
                    platform=platform,
                    approach=approach,
                    dop="serial", 
                    target="cuda", 
                    devices=v_devices,
                    gpus=v_gpus, 
                    r_nparallel=1,
                    r_rpc_host="0.0.0.0",
                    r_rpc_port=9000,
                    r_rpc_port_end=9099,
                    meta={"gpus": k_gpus, "devices": k_devices}
                )
                save(os.path.join("gpus_" + k_gpus, d), j, fname)
                
generate_multi_gpu_rpc()