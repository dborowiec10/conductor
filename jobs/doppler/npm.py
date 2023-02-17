from generator.generate import get_job
import os
import json

base = os.path.dirname(os.path.realpath(__file__))

# Load prereqs 
autotuners = {}
with open(os.path.join(base, "..", "templates", "autotuners.json"), "r") as f:
    autotuners = json.load(f)

workloads_tp = {}
with open(os.path.join(base, "..", "templates", "workloads_tp.json"), "r") as f:
    workloads_tp = json.load(f)

workloads_mod = {}
with open(os.path.join(base, "..", "templates", "workloads_mod.json"), "r") as f:
    workloads_mod = json.load(f)

base_save_path = os.path.join(base, "npm")

settings = {
    "num_trials": 500,
    "batch_size": 64,
    "early_stop": None
}
devices = ["localmachine:v100.0"]
gpus = [0]
platform = "intelv100"
dop = 2
approach = "npm"

def save(dir, job, filename):
    dirpath = os.path.join(base_save_path, dir)
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w") as f:
        json.dump(job, f, indent=4)
            
# tensor programs cuda
for p in [2, 4, 8, 16, 32, 64, 128]:
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
                r_nparallel=p
            ) 
            save(os.path.join("p" + str(p), d), j, fname)
        
# models cuda
for p in [2, 4, 8, 16, 32, 64, 128]:
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
                r_nparallel=p
            )
            save(os.path.join("p" + str(p), d), j, fname)