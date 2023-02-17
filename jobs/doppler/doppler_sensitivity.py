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
        {"name": "sh_35", "type": "tensor_program", "identifier": "dense_small_batch_21b6da02.cuda"},

    ]
}

    
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

base_save_path = os.path.join(base, "doppler_sensitivity")

def save(dir, job, filename):
    dirpath = os.path.join(base_save_path, dir)
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w") as f:
        json.dump(job, f, indent=4)

# rpc jobs

settings = {
    "num_trials": 500,
    "batch_size": 64,
    "early_stop": None
}

devices = ["localmachine:v100.0"]
gpus = [0]
platform = "intelv100"
dop = "doppler"
approach = "doppler_sensitivity"


def generate_multiple_batch():
    workloads_tp_multibatch = {
        "bs_1": {
            "cuda": [
                {"name": "sh_1", "type": "tensor_program", "identifier": "convnext_large_1"},
                {"name": "sh_3", "type": "tensor_program", "identifier": "convnext_large_3"},
                {"name": "sh_4", "type": "tensor_program", "identifier": "convnext_large_4"},
                {"name": "sh_6", "type": "tensor_program", "identifier": "convnext_large_6"},
                {"name": "sh_27", "type": "tensor_program", "identifier": "convnext_large_27"},
            ]
        },
        "bs_2": {
            "cuda": [
                {"name": "sh_1", "type": "tensor_program", "identifier": "convnext_large_bs2_1"},
                {"name": "sh_3", "type": "tensor_program", "identifier": "convnext_large_bs2_3"},
                {"name": "sh_4", "type": "tensor_program", "identifier": "convnext_large_bs2_4"},
                {"name": "sh_6", "type": "tensor_program", "identifier": "convnext_large_bs2_6"},
                {"name": "sh_27", "type": "tensor_program", "identifier": "convnext_large_bs2_27"},
            ]
        },
        "bs_4": {
            "cuda": [
                {"name": "sh_1", "type": "tensor_program", "identifier": "convnext_large_bs4_1"},
                {"name": "sh_3", "type": "tensor_program", "identifier": "convnext_large_bs4_3"},
                {"name": "sh_4", "type": "tensor_program", "identifier": "convnext_large_bs4_4"},
                {"name": "sh_6", "type": "tensor_program", "identifier": "convnext_large_bs4_6"},
                {"name": "sh_27", "type": "tensor_program", "identifier": "convnext_large_bs4_27"},
            ]
        },
        "bs_6": {
            "cuda": [
                {"name": "sh_1", "type": "tensor_program", "identifier": "convnext_large_bs6_1"},
                {"name": "sh_3", "type": "tensor_program", "identifier": "convnext_large_bs6_3"},
                {"name": "sh_4", "type": "tensor_program", "identifier": "convnext_large_bs6_4"},
                {"name": "sh_6", "type": "tensor_program", "identifier": "convnext_large_bs6_6"},
                {"name": "sh_27", "type": "tensor_program", "identifier": "convnext_large_bs6_27"},
            ]
        },
    }
    
    for kw, vw in workloads_tp_multibatch.items():
        for wptcuda in vw["cuda"]:
            for atk, atv in autotuner_doppler.items():
                d, j, fname = get_job(
                    int(wptcuda["name"].split("_")[-1]), 
                    wptcuda, 
                    atk, 
                    atv, 
                    settings,
                    platform=platform,
                    approach="doppler_sensitivity",
                    dop="doppler", 
                    target="cuda", 
                    devices=devices,
                    gpus=gpus, 
                    r_nparallel=64,
                    meta={"batch_size": kw}
                )
                save(os.path.join("batch_size", "doppler", kw, d), j, fname)
                
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
                    meta={"batch_size": kw}
                )
                save(os.path.join("batch_size", "serial", kw, d), j, fname)


def generate_many_trials():
    workloads_tp_many_trials = {
        "cuda": [
            {"name": "sh_0", "type": "tensor_program", "identifier": "batch_matmul_51268830.cuda"},
            {"name": "sh_19", "type": "tensor_program", "identifier": "conv2d_nchw_607f6363.cuda"},
            {"name": "sh_29", "type": "tensor_program", "identifier": "conv3d_ncdhw_59c7a972.cuda"},
        ]
    }
    setts = copy.deepcopy(settings)
    setts["num_trials"] = 2000
    for wptcuda in workloads_tp_many_trials["cuda"]:
        for atk, atv in autotuner_doppler.items():
            d, j, fname = get_job(
                int(wptcuda["name"].split("_")[-1]), 
                wptcuda, 
                atk, 
                atv, 
                setts,
                platform=platform,
                approach="doppler_sensitivity",
                dop="doppler", 
                target="cuda", 
                devices=devices,
                gpus=gpus, 
                r_nparallel=64,
                meta={"num_trials": 2000}
            )
            save(os.path.join("many_trials", "doppler", d), j, fname)
            
        for atk, atv in autotuner_serial.items():
            d, j, fname = get_job(
                int(wptcuda["name"].split("_")[-1]), 
                wptcuda, 
                atk, 
                atv, 
                setts,
                platform=platform,
                approach="serial",
                dop="serial", 
                target="cuda", 
                devices=devices,
                gpus=gpus, 
                r_nparallel=1,
                meta={"num_trials": 2000}
            )
            save(os.path.join("many_trials", "serial", d), j, fname)


def generate_less_repeats():
    workloads_tp_less_repeats = {
        "cuda": [
            {"name": "sh_0", "type": "tensor_program", "identifier": "batch_matmul_51268830.cuda"},
            {"name": "sh_19", "type": "tensor_program", "identifier": "conv2d_nchw_607f6363.cuda"},
            {"name": "sh_29", "type": "tensor_program", "identifier": "conv3d_ncdhw_59c7a972.cuda"},
        ]
    }
    for wptcuda in workloads_tp_less_repeats["cuda"]:
        for atk, atv in autotuner_doppler.items():
            d, j, fname = get_job(
                int(wptcuda["name"].split("_")[-1]), 
                wptcuda, 
                atk, 
                atv, 
                settings,
                platform=platform,
                approach="doppler_sensitivity",
                dop="doppler", 
                target="cuda", 
                devices=devices,
                gpus=gpus, 
                r_nparallel=64,
                num_avg_runs=3,
                num_measure_repeat=1,
                min_repeat_ms=100
            )
            save(os.path.join("less_repeats", "doppler", d), j, fname)
            
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
                num_avg_runs=3,
                num_measure_repeat=1,
                min_repeat_ms=100
            )
            save(os.path.join("less_repeats", "serial", d), j, fname)
    pass


def generate_basic_sensitivity_jobs():
    sensitivity = {
        "delta_threshold": [
            {"variant": "0_1", "mconf": {"delta_threshold": 0.1}, "r_timeout": None},
            {"variant": "0_15", "mconf": {"delta_threshold": 0.15}, "r_timeout": None},
            {"variant": "0_25", "mconf": {"delta_threshold": 0.25}, "r_timeout": None},
            {"variant": "0_50", "mconf": {"delta_threshold": 0.50}, "r_timeout": None}
        ],
        "perc_remeasure": [
            {"variant": "0_1", "mconf": {"perc_remeasure_samples": 0.1}, "r_timeout": None},
            {"variant": "0_15", "mconf": {"perc_remeasure_samples": 0.15}, "r_timeout": None},
            {"variant": "0_30", "mconf": {"perc_remeasure_samples": 0.30}, "r_timeout": None},
            {"variant": "0_50", "mconf": {"perc_remeasure_samples": 0.50}, "r_timeout": None},
            {"variant": "1_sampl", "mconf": {"fixed_samples": True, "fixed_sample_count": 1}, "r_timeout": None}
        ],
        "rank_remeasure": [
            {"variant": "0_010", "mconf": {"rank_remeasure_perc": 0.010}, "r_timeout": None},
            {"variant": "0_050", "mconf": {"rank_remeasure_perc": 0.050}, "r_timeout": None},
            {"variant": "0_075", "mconf": {"rank_remeasure_perc": 0.075}, "r_timeout": None},
            {"variant": "0_150", "mconf": {"rank_remeasure_perc": 0.150}, "r_timeout": None},
            {"variant": "0_200", "mconf": {"rank_remeasure_perc": 0.200}, "r_timeout": None}
        ],
        "timeout": [
            {"variant": "2", "mconf": {"fixed_timeout": True}, "r_timeout": 2},
            {"variant": "3", "mconf": {"fixed_timeout": True}, "r_timeout": 3},
            {"variant": "5", "mconf": {"fixed_timeout": True}, "r_timeout": 5},
            {"variant": "10", "mconf": {"fixed_timeout": True}, "r_timeout": 10},
            {"variant": "20", "mconf": {"fixed_timeout": True}, "r_timeout": 20},
            {"variant": "30", "mconf": {"fixed_timeout": True}, "r_timeout": 30}
        ]
    }
    # generate basic sensitivity jobs
    for sk, sv in sensitivity.items():
        for svoption in sv:
            for wtpcuda in workloads_tp["cuda"]:
                for atk, atv in autotuner_doppler.items():
                    if svoption["r_timeout"] is not None:
                        tout = svoption["r_timeout"]
                    else:
                        tout = 3
                    d, j, fname = get_job(
                        int(wtpcuda["name"].split("_")[-1]), 
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
                        r_nparallel=64,
                        r_timeout=tout,
                        measuerer_config=svoption["mconf"],
                        meta={
                            "sensitivity_kind": sk,
                            "sensitivity_variant": svoption["variant"],
                            "sensitivity_measurer_config": svoption["mconf"],
                            "sensitivity_r_timeout": svoption["r_timeout"],
                        }
                    )
                    save(os.path.join(sk, d, svoption["variant"]), j, fname)
                    
generate_basic_sensitivity_jobs()
generate_multiple_batch()
generate_many_trials()
generate_less_repeats()