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

workloads_tp_bert = {
    "cuda": [
        {"name": "sh_0", "type": "tensor_program", "identifier": "bert_0"},
        {"name": "sh_1", "type": "tensor_program", "identifier": "bert_1"},
        {"name": "sh_2", "type": "tensor_program", "identifier": "bert_2"},
        {"name": "sh_3", "type": "tensor_program", "identifier": "bert_3"},
        {"name": "sh_4", "type": "tensor_program", "identifier": "bert_4"},
        {"name": "sh_5", "type": "tensor_program", "identifier": "bert_5"},
        {"name": "sh_6", "type": "tensor_program", "identifier": "bert_6"},
        {"name": "sh_7", "type": "tensor_program", "identifier": "bert_7"},
        {"name": "sh_8", "type": "tensor_program", "identifier": "bert_8"},
        {"name": "sh_9", "type": "tensor_program", "identifier": "bert_9"},
        {"name": "sh_10", "type": "tensor_program", "identifier": "bert_10"}
    ]
}


base_save_path = os.path.join(base, "doppler_bert")

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
        
        
def generate_bert():
    for wptcuda in workloads_tp_bert["cuda"]:
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

generate_bert()