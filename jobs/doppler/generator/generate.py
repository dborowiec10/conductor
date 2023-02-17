import json
import os
import copy

base = os.path.dirname(os.path.realpath(__file__))
    
configs = {
    "builder": {},
    "cost_model": {},
    "evaluator": {},
    "filter": {},
    "measurer": {},
    "method": {},
    "optimizer": {},
    "profiler": {},
    "runner": {},
    "sampler": {},
    "search_policy": {},
    "strategy": {}
}

with open(os.path.join(base, "..", "..", "templates", "builder.json"), "r") as f:
    configs["builder"] = json.load(f)
    
with open(os.path.join(base, "..", "..", "templates", "cost_model.json"), "r") as f:
    configs["cost_model"] = json.load(f)

with open(os.path.join(base, "..", "..", "templates", "evaluator.json"), "r") as f:
    configs["evaluator"] = json.load(f)

with open(os.path.join(base, "..", "..", "templates", "filter.json"), "r") as f:
    configs["filter"] = json.load(f)
    
with open(os.path.join(base, "..", "..", "templates", "measurer.json"), "r") as f:
    configs["measurer"] = json.load(f)
    
with open(os.path.join(base, "..", "..", "templates", "method.json"), "r") as f:
    configs["method"] = json.load(f)
    
with open(os.path.join(base, "..", "..", "templates", "optimizer.json"), "r") as f:
    configs["optimizer"] = json.load(f)

with open(os.path.join(base, "..", "..", "templates", "profiler.json"), "r") as f:
    configs["profiler"] = json.load(f)

with open(os.path.join(base, "..", "..", "templates", "runner.json"), "r") as f:
    configs["runner"] = json.load(f)

with open(os.path.join(base, "..", "..", "templates", "sampler.json"), "r") as f:
    configs["sampler"] = json.load(f)

with open(os.path.join(base, "..", "..", "templates", "search_policy.json"), "r") as f:
    configs["search_policy"] = json.load(f)
    
with open(os.path.join(base, "..", "..", "templates", "strategy.json"), "r") as f:
    configs["strategy"] = json.load(f)


# generate job metadata
def get_metadata(platform, autotuner, model, dop="serial", approach="serial", attach=None):
    retval = {
        "platform": platform,
        "autotuner": autotuner,
        "approach": approach,
        "model": model,
        "dop": dop,
        "run_idx": 0
    }
    if attach is not None:
        retval.update(attach)    
    return retval

# returns: inputs list, internal input name, identifier, internal output name
def get_inputs(inpdef):
    if inpdef["type"] == "tensor_program":
        dct = [inpdef]
        return dct, inpdef["identifier"], inpdef["name"], "t_" + inpdef["name"]
    else:
        dct = [inpdef]
        return dct, inpdef["model_name"], inpdef["name"], "t_" + inpdef["name"]
    
def get_task(inp, out, specs, configs, profilers):
    return {
        "name": "test_task",
        "type": "orchestrator:default",
        "inputs": [inp],
        "outputs": [out],
        "specifications": specs,
        "configurations": configs,
        "profilers": profilers
    }
    
def get_orchestrator_spec(
    at, 
    settings, 
    target="cuda", 
    devices=["localmachine:0"],
    mps_devices=[], 
    b_nparallel=None, 
    r_nparallel=1,
    num_avg_runs=20, 
    num_measure_repeat=3, 
    min_repeat_ms=100,
    r_timeout=3,
    r_rpc_host=None,
    r_rpc_port=None,
    r_rpc_port_end=None,
    measurer_conf=None
):
    spec = {
        "name": "orchestrator_default_spec",
        "type": "orchestrator",
        "spec": {
            "measurer": at["measurer"],
            "builder": at["builder"],
            "runner": at["runner"],
            "evaluator": at["evaluator"]
        },
        "settings": settings
    }
    measurer_config = copy.deepcopy(configs["measurer"][at["measurer"]])
    measurer_config["target"] = target
    measurer_config["devices"] = devices
    measurer_config["nvidia_mps_devices"] = mps_devices
    if measurer_conf is not None:
        measurer_config.update(measurer_conf)
        
    evaluator_config = copy.deepcopy(configs["evaluator"][at["evaluator"]])
    evaluator_config["num_avg_runs"] = evaluator_config["num_avg_runs"] if evaluator_config["num_avg_runs"] == num_avg_runs else num_avg_runs
    evaluator_config["num_measure_repeat"] = evaluator_config["num_measure_repeat"] if evaluator_config["num_measure_repeat"] == num_measure_repeat else num_measure_repeat
    evaluator_config["min_repeat_ms"] = evaluator_config["min_repeat_ms"] if evaluator_config["min_repeat_ms"] == min_repeat_ms else min_repeat_ms
    
    runner_config = copy.deepcopy(configs["runner"][at["runner"]])
    runner_config["n_parallel"] = r_nparallel
    runner_config["timeout"] = r_timeout
    if r_rpc_host is not None:
        runner_config["rpc_host"] = r_rpc_host
    if r_rpc_port is not None:
        runner_config["rpc_port"] = r_rpc_port
    if r_rpc_port_end is not None:
        runner_config["rpc_port_end"] = r_rpc_port_end
    
    builder_config = copy.deepcopy(configs["builder"][at["builder"]])
    if at["builder"] == "parallel":
        builder_config["n_parallel"] = b_nparallel
    
    config_ids = [
        "orchestrator:default.1",
        "measurer:" + at["measurer"] + ".1",
        "builder:" + at["builder"] + ".1",
        "runner:" + at["runner"] + ".1",
        "evaluator:" + at["evaluator"] + ".1"
    ]
    cnfgs = [
        {
            "entity": "orchestrator:default",
            "name": "orchestrator:default.1",
            "configuration": {}  
        },
        {
            "entity": "measurer:" + at["measurer"],
            "name": "measurer:" + at["measurer"] + ".1",
            "configuration": measurer_config
        },
        {
            "entity": "builder:" + at["builder"],
            "name": "builder:" + at["builder"] + ".1",
            "configuration": builder_config
        },
        {
            "entity": "runner:" + at["runner"],
            "name": "runner:" + at["runner"] + ".1",
            "configuration": runner_config
        },
        {
            "entity": "evaluator:" + at["evaluator"],
            "name": "evaluator:" + at["evaluator"] + ".1",
            "configuration": evaluator_config
        }
    ]
    return spec, "orchestrator_default_spec", cnfgs, config_ids
    
def get_inp_meth_map_spec(inptype, inp, out, at, method_id):
    cnfgs = [{
        "entity": "strategy:" + at["strategy"],
        "name": "strategy:" + at["strategy"] + ".1",
        "configuration": copy.deepcopy(configs["strategy"][at["strategy"]])
    }]
    config_ids = ["strategy:" + at["strategy"] + ".1"]
    mp = {
        "name": "test_input_method_map",
        "type": "input_method_map",
        "spec": {
            "strategy": at["strategy"],
            "group_type": inptype,
            "maps": [{
                "input": inp,
                "output": out,
                "method": method_id
            }]
        }
    }
    mp_id = "test_input_method_map"
    return mp, mp_id, cnfgs, config_ids

def get_method_spec(at):
    method_identifier = at["method_type"]
    scheduling = None
    kind = None
    if "standalone" in at["method_type"]:
        kind = "standalone"
        method_identifier = method_identifier + "_" + at["method_name"]
        if "template" in at["method_type"]:
            scheduling = "template"
        elif "flex" in at["method_type"]:
            scheduling = "flex"  
    elif "composite" in at["method_type"]:
        kind = "composite"
        method_identifier = method_identifier + "_" + at["cost_model"]
        if "sketch" in at["method_type"]:
            scheduling = "sketch"
            method_identifier = method_identifier + "_" + at["search_policy"]
        elif "template" in at["method_type"]:
            scheduling = "template"
            method_identifier = method_identifier + "_" + at["optimizer"]
            if at["sampler"] is not None:
                method_identifier = method_identifier + "_s_" + at["sampler"]
            if at["filter"] is not None:
                method_identifier = method_identifier + "_f_" + at["filter"]
            else:
                method_identifier = method_identifier + "_f_default"
    method_spec = {
        "name": "meth_" + method_identifier,
        "type": "method",
        "spec": {
            "scheduling": scheduling,
            "kind": kind,
            "method_name": at["method_name"],
            "cost_model": at["cost_model"],
            "optimizer": at["optimizer"],
            "search_policy": at["search_policy"],
            "sampler": at["sampler"],
            "filter": at["filter"]
        },
        "settings": {}
    }
    meth_conf = copy.deepcopy(configs["method"][scheduling][kind])
    if kind == "standalone":
        meth_conf = meth_conf[at["method_name"]]
    entity = "method:" + scheduling + ":" + kind
    if at["method_name"] is not None:
        entity = entity + ":" + at["method_name"]
    cnfgs = [{"entity": entity, "name": entity + ".1", "configuration": meth_conf}]
    config_ids = [entity + ".1"]
    if at["cost_model"]:
        cnfgs.append({
            "entity": "cost_model:" + at["cost_model"],
            "name": "cost_model:" + at["cost_model"] + ".1",
            "configuration": copy.deepcopy(configs["cost_model"][scheduling][at["cost_model"]])
        })
        config_ids.append("cost_model:" + at["cost_model"] + ".1")
    if at["optimizer"]:
        cnfgs.append({
            "entity": "optimizer:" + at["optimizer"],
            "name": "optimizer:" + at["optimizer"] + ".1",
            "configuration": copy.deepcopy(configs["optimizer"][scheduling][at["optimizer"]])
        })
        config_ids.append("optimizer:" + at["optimizer"] + ".1")
    if at["search_policy"]:
        cnfgs.append({
            "entity": "search_policy:" + at["search_policy"],
            "name": "search_policy:" + at["search_policy"] + ".1",
            "configuration": copy.deepcopy(configs["search_policy"][scheduling][at["search_policy"]])
        })
        config_ids.append("search_policy:" + at["search_policy"] + ".1")
    if at["sampler"]:
        cnfgs.append({
            "entity": "sampler:" + at["sampler"],
            "name": "sampler:" + at["sampler"] + ".1",
            "configuration": copy.deepcopy(configs["sampler"][scheduling][at["sampler"]])
        })
        config_ids.append("sampler:" + at["sampler"] + ".1")
    if at["filter"]:
        cnfgs.append({
            "entity": "filter:" + at["filter"],
            "name": "filter:" + at["filter"] + ".1",
            "configuration": copy.deepcopy(configs["filter"][scheduling][at["filter"]])
        })
        config_ids.append("filter:" + at["filter"] + ".1")
    return method_spec, method_identifier, "meth_" + method_identifier, cnfgs, config_ids

def get_profilers(gpus=[0]):
    spec = [
        {"type": "time_profiler",
         "scope": "strategy",
         "configuration": "time_profiler.1",
         "collection": "strategy_runs"},
        {"type": "time_profiler",
         "scope": "method",
         "configuration": "time_profiler.2",
         "collection": "method_runs"},
        {"type": "time_profiler",
         "scope": "builder",
         "configuration": "time_profiler.3",
         "collection": "compilations"},
        {"type": "time_profiler",
         "scope": "runner",
         "configuration": "time_profiler.4",
         "collection": "measurements"},
        {"type": "system_monitor",
         "scope": "strategy",
         "configuration": "system_monitor.1",
         "collection": "sysmons"}
    ]
    sysmonconf = copy.deepcopy(configs["profiler"]["system_monitor"])
    sysmonconf["gpus"] = gpus
    if len(gpus) <= 0:
        sysmonconf["metrics"]["gpu_mem_free"] = False
        sysmonconf["metrics"]["gpu_mem_used"] = False
        sysmonconf["metrics"]["gpu_mem_total"] = False
        sysmonconf["metrics"]["gpu_util"] = False
        sysmonconf["metrics"]["gpu_mem_util"] = False
        sysmonconf["metrics"]["gpu_power"] = False
        sysmonconf["metrics"]["gpu_clock_sm"] = False
        sysmonconf["metrics"]["gpu_clock_mem"] = False
    cnfgs = [
        {"entity": "time_profiler",
         "name": "time_profiler.1",
         "configuration": None},
        {
         "entity": "time_profiler",
         "name": "time_profiler.2",
         "configuration": None},
        {
         "entity": "time_profiler",
         "name": "time_profiler.3",
         "configuration": None},
        {
         "entity": "time_profiler",
         "name": "time_profiler.4",
         "configuration": None
        },
        {"entity": "system_monitor",
         "name": "system_monitor.1",
         "configuration": sysmonconf}
    ]
    config_ids = [
        "time_profiler.1",
        "time_profiler.2",
        "time_profiler.3",
        "time_profiler.4",
        "system_monitor.1"
    ]
    return spec, cnfgs, config_ids

def get_job(
    idx, wrkld, atk, atv, settings, 
    target="cuda", 
    platform="intelv100", 
    dop="serial", 
    approach="serial", 
    gpus=[0], 
    devices=["localmachine:v100.0"],
    mps_devices=[],
    r_nparallel=1,
    r_timeout=3,
    r_rpc_host=None,
    r_rpc_port=None,
    r_rpc_port_end=None,
    num_avg_runs=20, 
    num_measure_repeat=3, 
    min_repeat_ms=100,
    meta=None,
    measuerer_config=None
):
    if wrkld["type"] == "model":
        meta = get_metadata(
            platform, 
            atk, 
            wrkld["model_name"], 
            dop=dop, 
            approach=approach,
            attach=meta
        )
    else:
        meta = get_metadata(
            platform, 
            atk, 
            None, 
            dop=dop, 
            approach=approach,
            attach=meta
        )

    inp, ident, intern_inp_name, intern_out_name = get_inputs(wrkld)
    ospec, ospec_id, ospec_configs, ospec_config_ids = get_orchestrator_spec(
        atv, 
        settings, 
        devices=devices, 
        mps_devices=mps_devices, 
        r_nparallel=r_nparallel,
        r_timeout=r_timeout,
        r_rpc_host=r_rpc_host,
        r_rpc_port=r_rpc_port,
        r_rpc_port_end=r_rpc_port_end,
        measurer_conf=measuerer_config,
        num_avg_runs=num_avg_runs,
        num_measure_repeat=num_measure_repeat,
        min_repeat_ms=min_repeat_ms
    )
    mspec, dir_name, mspec_id, mspec_configs, mspec_config_ids = get_method_spec(atv)
    mpspec, mpspec_id, mpspec_configs, mpspec_config_ids = get_inp_meth_map_spec(
        wrkld["type"], 
        intern_inp_name, 
        intern_out_name, 
        atv, 
        mspec_id
    )
    profspec, prof_configs, prof_config_ids = get_profilers(gpus=gpus)
    if wrkld["type"] == "tensor_program":
        fname = ("%.2d" % idx) + "_" + ident + "_" + dir_name + "_" + target
    else:
        fname = ident + "_" + dir_name + "_" + target
    dir = wrkld["type"] + "s_" + target + "_" + dir_name
    job = {
        "job_name_prefix": fname,
        "metadata": meta,
        "inputs": inp,
        "specifications": [ospec, mspec, mpspec],
        "task": get_task(
            intern_inp_name, 
            intern_out_name,
            [ospec_id, mspec_id, mpspec_id],
            ospec_config_ids + mspec_config_ids + mpspec_config_ids + prof_config_ids,
            profspec
        ),
        "configurations": ospec_configs + mspec_configs + mpspec_configs + prof_configs,
    }
    return dir, job, fname + ".json"




