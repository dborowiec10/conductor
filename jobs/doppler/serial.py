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

settings = {
    "num_trials": 500,
    "batch_size": 64,
    "early_stop": None
}

base_save_path = os.path.join(base, "serial")

cuda_devices = ["localmachine:v100.0"]
llvm_devices = ["localmachine:cpu.0"]

gpus = [0]
platform = "intelv100"
dop = "serial"
approach = "serial"

def save(dir, job, filename):
    dirpath = os.path.join(base_save_path, dir)
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w") as f:
        json.dump(job, f, indent=4)

# tensor programs llvm
for k, wtpxllvm in enumerate(workloads_tp["llvm"]):
    for atk, atv in autotuners.items():
        d, j, fname = get_job(
            k, 
            wtpxllvm, 
            atk, 
            atv, 
            settings,
            platform=platform,
            approach=approach,
            dop=dop, 
            target="llvm", 
            devices=llvm_devices,
            r_nparallel=1,
            gpus=[]
        )
        save(d, j, fname)
        
# tensor programs cuda
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
            devices=cuda_devices,
            r_nparallel=1,
            gpus=gpus
        ) 
        save(d, j, fname)
        
# models llvm 
for k, wmodllvm in enumerate(workloads_mod["llvm"]):
    for atk, atv in autotuners.items():
        d, j, fname = get_job(
            k, 
            wmodllvm, 
            atk, 
            atv, 
            settings,
            platform=platform,
            approach=approach,
            dop=dop, 
            target="llvm", 
            devices=llvm_devices,
            r_nparallel=1,
            gpus=[]
        )
        save(d, j, fname)
        
# models cuda
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
            devices=cuda_devices,
            r_nparallel=1,
            gpus=gpus
        )
        save(d, j, fname)