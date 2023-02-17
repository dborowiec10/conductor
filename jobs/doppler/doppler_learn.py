from generator.generate import get_job
import os
import json

base = os.path.dirname(os.path.realpath(__file__))

workloads_tp = {}
with open(os.path.join(base, "..", "templates", "workloads_tp.json"), "r") as f:
    workloads_tp = json.load(f)

workloads_mod = {}
with open(os.path.join(base, "..", "templates", "workloads_mod.json"), "r") as f:
    workloads_mod = json.load(f)

autotuners = {
    "chameleon": {   
        "orchestrator": "default", 
        "strategy": "sequential",
        "method_type": "composite_template",
        "method_name": None,           
        "cost_model": "tvm_xgb_itervar_reg",
        "optimizer": "reinforcement_learning", 
        "search_policy": None,           
        "sampler": "adaptive", 
        "filter": "default",        
        "measurer": "doppler_learn", 
        "evaluator": "activity",  
        "builder": "parallel", 
        "runner": "local"
    },
    "autotvm": {   
        "orchestrator": "default", 
        "strategy": "sequential",
        "method_type": "composite_template",
        "method_name": None,           
        "cost_model": "tvm_xgb_itervar_rank",
        "optimizer": "simulated_annealing",    
        "search_policy": None,           
        "sampler": None,       
        "filter": "default",         
        "measurer": "doppler_learn", 
        "evaluator": "activity",  
        "builder": "parallel", 
        "runner": "local"
    },
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
        "measurer": "doppler_learn", 
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
        "measurer": "doppler_learn", 
        "evaluator": "activity",  
        "builder": "parallel", 
        "runner": "local"
    }
}
settings = {
    "num_trials": 500,
    "batch_size": 64,
    "early_stop": None
}

base_save_path = os.path.join(base, "doppler_learn")

devices = ["localmachine:v100.0"]
gpus = [0]
platform = "intelv100"
dop = "doppler"
approach = "doppler_learn"

def save(dir, job, filename):
    dirpath = os.path.join(base_save_path, dir)
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w") as f:
        json.dump(job, f, indent=4)
        
for k, wtpcuda in enumerate(workloads_tp["cuda"]):
    for atk, atv in autotuners.items():
        d, j, fname = get_job(
            k, 
            wtpcuda, 
            atk, 
            atv, 
            settings,
            platform=platform,
            approach=approach,
            dop=dop, 
            target="cuda", 
            devices=devices,
            gpus=gpus, 
            r_nparallel=64
        ) 
        save(d, j, fname)
        
for k, wmodcuda in enumerate(workloads_mod["cuda"]):
    for atk, atv in autotuners.items():
        d, j, fname = get_job(
            k, 
            wmodcuda, 
            atk, 
            atv, 
            settings, 
            platform=platform,
            approach=approach,
            dop=dop, 
            target="cuda", 
            devices=devices,
            gpus=gpus,
            r_nparallel=64
        )
        save(d, j, fname)