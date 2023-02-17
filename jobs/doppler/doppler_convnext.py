from generator.generate import get_job
import os
import json
import copy

base = os.path.dirname(os.path.realpath(__file__))
    
autotuner_doppler = {
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
        "measurer": "doppler_bic", 
        "evaluator": "activity",  
        "builder": "parallel", 
        "runner": "local"
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
        "measurer": "doppler_bic", 
        "evaluator": "activity",  
        "builder": "parallel", 
        "runner": "local"
    },
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
        "runner": "local"
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
        "runner": "local"
    },
}

workloads_tp_convnext = {
    "cuda": [
        {"name": "sh_0", "type": "tensor_program", "identifier": "convnext_large_0"},
        {"name": "sh_1", "type": "tensor_program", "identifier": "convnext_large_1"},
        {"name": "sh_2", "type": "tensor_program", "identifier": "convnext_large_2"},
        {"name": "sh_3", "type": "tensor_program", "identifier": "convnext_large_3"},
        {"name": "sh_4", "type": "tensor_program", "identifier": "convnext_large_4"},
        {"name": "sh_5", "type": "tensor_program", "identifier": "convnext_large_5"},
        {"name": "sh_6", "type": "tensor_program", "identifier": "convnext_large_6"},
        {"name": "sh_7", "type": "tensor_program", "identifier": "convnext_large_7"},
        {"name": "sh_8", "type": "tensor_program", "identifier": "convnext_large_8"},
        {"name": "sh_9", "type": "tensor_program", "identifier": "convnext_large_9"},
        {"name": "sh_10", "type": "tensor_program", "identifier": "convnext_large_10"},
        {"name": "sh_11", "type": "tensor_program", "identifier": "convnext_large_11"},
        {"name": "sh_12", "type": "tensor_program", "identifier": "convnext_large_12"},
        {"name": "sh_13", "type": "tensor_program", "identifier": "convnext_large_13"},
        {"name": "sh_14", "type": "tensor_program", "identifier": "convnext_large_14"},
        {"name": "sh_15", "type": "tensor_program", "identifier": "convnext_large_15"},
        {"name": "sh_16", "type": "tensor_program", "identifier": "convnext_large_16"},
        {"name": "sh_17", "type": "tensor_program", "identifier": "convnext_large_17"},
        {"name": "sh_18", "type": "tensor_program", "identifier": "convnext_large_18"},
        {"name": "sh_19", "type": "tensor_program", "identifier": "convnext_large_19"},
        {"name": "sh_20", "type": "tensor_program", "identifier": "convnext_large_20"},
        {"name": "sh_21", "type": "tensor_program", "identifier": "convnext_large_21"},
        {"name": "sh_22", "type": "tensor_program", "identifier": "convnext_large_22"},
        {"name": "sh_23", "type": "tensor_program", "identifier": "convnext_large_23"},
        {"name": "sh_24", "type": "tensor_program", "identifier": "convnext_large_24"},
        {"name": "sh_25", "type": "tensor_program", "identifier": "convnext_large_25"},
        {"name": "sh_26", "type": "tensor_program", "identifier": "convnext_large_26"},
        {"name": "sh_27", "type": "tensor_program", "identifier": "convnext_large_27"}
    ]
}

base_save_path = os.path.join(base, "doppler_convnext")

settings = {
    "num_trials": 500,
    "batch_size": 64,
    "early_stop": None
}

devices = ["localmachine:v100.0"]
gpus = [0]
platform = "intelv100"
dop = "doppler"
approach = "doppler"

def save(dir, job, filename):
    dirpath = os.path.join(base_save_path, dir)
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w") as f:
        json.dump(job, f, indent=4)

def generate_convnext():
    for wptcuda in workloads_tp_convnext["cuda"]:
        for atk, atv in autotuner_doppler.items():
            d, j, fname = get_job(
                int(wptcuda["name"].split("_")[-1]), 
                wptcuda, 
                atk, 
                atv, 
                settings,
                platform=platform,
                approach="doppler",
                dop="doppler", 
                target="cuda", 
                devices=devices,
                gpus=gpus, 
                r_nparallel=64,
            )
            save(os.path.join("doppler", d), j, fname)
        
        for atk, atv in autotuner_serial.items():
            d, j, fname = get_job(
                int(wptcuda["name"].split("_")[-1]), 
                wptcuda, 
                atk, 
                atv, 
                settings,
                platform=platform,
                approach="serial",
                dop="serial", 
                target="cuda", 
                devices=devices,
                gpus=gpus, 
                r_nparallel=1,
            )
            save(os.path.join("serial", d), j, fname)

generate_convnext()