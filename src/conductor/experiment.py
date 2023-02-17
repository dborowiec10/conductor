from conductor.store import Store
from conductor.mediation import ERROR_TYPES
import uuid
import time

class Experiment(object):
    _name = "experiment"
    current = None

    def __repr__(self):
        return Experiment._name

    def get_input_type(self, spec):
        _type = set()
        for inp in spec.inputs:
            _type.add(inp._type)
        _type = list(_type)
        if len(_type) > 1:
            return "mixed"
        else:
            return _type[0]

    def __init__(self, store_config, job_name, job_spec, override=False):
        self.store_config = store_config
        self.s = Store(
            store_config["host"],
            store_config["port"], 
            store_config["user"],
            store_config["pass"],
            store_config["db"]
        )
        if not override:
            self.exp = {
                "id": str(uuid.uuid4()),
                "name": job_name,
                "start": None,
                "end": None,
                "duration": None,
                "completed": False,
                "status": "started",
                "error": None,
                "type": job_spec.task._type,
                "input_type": self.get_input_type(job_spec),
                "metadata": job_spec.meta,
                "data": {}
            }
        else:
            self.exp = None
        self.current_stage = None

    def sync(self):
        self.exp = self.s.get("experiments", {"id": self.exp["id"]})

    def insert_one(self, coll, datum):
        base = self.get_base()
        base.update(datum)
        self.s.insert_one(coll, base)

    def insert_many(self, coll, data):
        if data != None and type(data) == list and len(data) > 0:
            dcopy = data.copy()
            base = self.get_base()
            for k, _ in enumerate(dcopy):
                dcopy[k].update(base)
            self.s.insert_many(coll, dcopy)

    def get_base(self):
        raise NotImplementedError()

    def __enter__(self):
        self.exp["start"] = time.time()
        self.exp_id = self.s.insert_one("experiments", self.exp)
        Experiment.current = self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.exp["end"] = time.time()
        self.exp["duration"] = self.exp["end"] - self.exp["start"]
        if exc_type is None:
            self.exp["status"] = "completed"
            self.exp["completed"] = True
        else:
            self.exp["status"] = "failed"
            self.exp["completed"] = False
            self.exp["error"] = {
                "type": str(exc_type),
                "value": str(exc_value),
                "trcbck": str(exc_traceback)
            }
        self.s.update_one("experiments", {"id": self.exp["id"]}, {
            "end": self.exp["end"],
            "duration": self.exp["duration"],
            "status": self.exp["status"],
            "completed": self.exp["completed"],
            "error": self.exp["error"]
        })

class TensorProgramCompilerExperiment(Experiment):
    _name = "tensor_program_compiler_experiment"

    def __repr__(self):
        return Experiment.__repr__(self) + ":" + TensorProgramCompilerExperiment._name

    def __init__(self, store_config, job_name, job_spec):
        Experiment.__init__(self, store_config, job_name, job_spec)
        self.exp["data"]["tensor_programs"] = []
        self.exp["data"]["options"] = []
        self.exp["data"]["config"] = None
        self.current_idx = -1

        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data": self.exp["data"]   
            }
        )

    def get_base(self):
        return {
            "experiment_id": self.exp["id"],
            "tensor_program": self.exp["data"]["tensor_programs"][self.current_idx],
            "options": self.exp["data"]["options"][self.current_idx],
        }

    def set_index(self, idx):
        self.current_idx = idx

    def set_config(self, config):
        self.exp["data"]["config"] = config
        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data.config": self.exp["data"]["config"],
            }
        )

    def add_tp_and_opt(self, tp, options):
        self.exp["data"]["tensor_programs"].append(tp)
        self.exp["data"]["options"].append(options)
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.tensor_programs": self.exp["data"]["tensor_programs"],
                "data.options": self.exp["data"]["options"]
            }
        )

class ModelCompilerExperiment(Experiment):
    _name = "model_compiler_experiment"

    def __repr__(self):
        return Experiment.__repr__(self) + ":" + ModelCompilerExperiment._name

    def __init__(self, store_config, job_name, job_spec):
        Experiment.__init__(self, store_config, job_name, job_spec)
        self.exp["data"]["models"] = []
        self.exp["data"]["options"] = []
        self.exp["data"]["config"] = None
        self.current_idx = -1

        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data": self.exp["data"]
            }
        )

    def get_base(self):
        return {
            "experiment_id": self.exp["id"],
            "model": self.exp["data"]["models"][self.current_idx],
            "options": self.exp["data"]["options"][self.current_idx],
        }

    def set_index(self, idx):
        self.current_idx = idx

    def set_config(self, config):
        self.exp["data"]["config"] = config
        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data.config": self.exp["data"]["config"],
            }
        )

    def add_md_and_opt(self, tp, options):
        self.exp["data"]["models"].append(tp)
        self.exp["data"]["options"].append(options)
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.models": self.exp["data"]["models"],
                "data.options": self.exp["data"]["options"]
            }
        )

class TensorProgramExecutorExperiment(Experiment):
    _name = "tensor_program_executor_experiment"

    def __repr__(self):
        return Experiment.__repr__(self) + ":" + TensorProgramExecutorExperiment._name

    def __init__(self, store_config, job_name, job_spec):
        Experiment.__init__(self, store_config, job_name, job_spec)
        self.exp["data"]["tensor_programs"] = []
        self.exp["data"]["options"] = []
        self.exp["data"]["config"] = None
        self.current_idx = -1

        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data": self.exp["data"]
            }
        )

    def get_base(self):
        return {
            "experiment_id": self.exp["id"],
            "tensor_program": self.exp["data"]["tensor_programs"][self.current_idx],
            "options": self.exp["data"]["options"][self.current_idx],
        }

    def set_index(self, idx):
        self.current_idx = idx

    def set_config(self, config):
        self.exp["data"]["config"] = config
        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data.config": self.exp["data"]["config"]
            }
        )

    def add_tp_and_opt(self, tp, options):
        self.exp["data"]["tensor_programs"].append(tp)
        self.exp["data"]["options"].append(options)
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.tensor_programs": self.exp["data"]["tensor_programs"],
                "data.options": self.exp["data"]["options"]
            }
        )

class ModelExecutorExperiment(Experiment):
    _name = "model_executor_experiment"

    def __repr__(self):
        return Experiment.__repr__(self) + ":" + ModelExecutorExperiment._name

    def __init__(self, store_config, job_name, job_spec):
        Experiment.__init__(self, store_config, job_name, job_spec)
        self.exp["data"]["models"] = []
        self.exp["data"]["options"] = []
        self.exp["data"]["config"] = None
        self.current_idx = -1

        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data": self.exp["data"]
            }
        )

    def get_base(self):
        return {
            "experiment_id": self.exp["id"],
            "model": self.exp["data"]["models"][self.current_idx],
            "options": self.exp["data"]["options"][self.current_idx],
        }

    def set_index(self, idx):
        self.current_idx = idx

    def set_config(self, config):
        self.exp["data"]["config"] = config
        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data.config": self.exp["data"]["config"]
            }
        )

    def add_md_and_opt(self, tp, options):
        self.exp["data"]["models"].append(tp)
        self.exp["data"]["options"].append(options)
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.models": self.exp["data"]["models"],
                "data.options": self.exp["data"]["options"]
            }
        )

class OrchestratorExperiment(Experiment):
    _name = "orchestrator_experiment"
    
    @classmethod
    def from_sync(cls, store_config, exp_id):
        exp = cls(store_config, None, None, override=True)
        exp.exp = exp.s.get("experiments", {"id": exp_id})
        return exp
    
    def __repr__(self):
        return Experiment.__repr__(self) + ":" + OrchestratorExperiment._name

    def __init__(self, store_config, job_name, job_spec, override=False):
        Experiment.__init__(self, store_config, job_name, job_spec, override=override)

        if not override:
            self.exp["data"] = {
                "build_duration": 0,
                "run_duration": 0,
                "calc_duration": 0,
                "rank_duration": 0,
                "target": None,
                "target_host": None,
                "strategy": {
                    "name": None,
                    "setting": None,
                    "measurer": None,
                    "evaluator": None,
                    "builder": None,
                    "runner": None
                },
                "methods": [],
                "tasks": [],
                "all_logs_file_id": None,
                "best_logs_file_id": None,
                "rank_logs_file_id": None,
                "best_rank_logs_file_id": None,
            }

        self.current_task = None
        self.current_method = None

    def read_file(self, file):
        with open(file, "rb") as fp:
            return fp.read()
    
    def get_task(self, t):
        exclude_keys = ['status']
        return {k: t[k] for k in set(list(t.keys())) - set(exclude_keys)}

    def get_base(self):
        return {
            "experiment_id": self.exp["id"],
            "task": self.get_task(self.current_task),
            "method": self.current_method
        }

    def set_task(self, idx):
        self.current_task = self.exp["data"]["tasks"][idx]

    def set_method(self, idx):
        self.current_method = self.exp["data"]["methods"][idx]

    def set_experiment_stage(self, stage):
        self.current_stage = stage

    def insert_logs(self, log_type, logs, filename):
        base = {
            "experiment_id": self.exp["id"],
            "task": self.get_task(self.current_task),
            "method": self.current_method,
            "input_type": self.exp["input_type"],
            "type": log_type
        }
        for k, _ in enumerate(logs):
            logs[k].update(base)
        self.insert_many("logs", logs)
        quickmap = {"all": "all_logs_file_id", "best": "best_logs_file_id", "rank": "rank_logs_file_id", "best_rank": "best_rank_logs_file_id"}
        self.exp[quickmap[log_type]] = self.s.gridfs_insert(self.read_file(filename))
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data." + quickmap[log_type]: self.exp["data"][quickmap[log_type]]
            }
        )

    def set_strategy(self, strategy_name, strategy_setting, measurer, evaluator, builder, runner):
        self.exp["data"]["strategy"]["name"] = strategy_name
        self.exp["data"]["strategy"]["setting"] = strategy_setting
        self.exp["data"]["strategy"]["measurer"] = measurer
        self.exp["data"]["strategy"]["evaluator"] = evaluator
        self.exp["data"]["strategy"]["builder"] = builder
        self.exp["data"]["strategy"]["runner"] = runner
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.strategy": self.exp["data"]["strategy"]
            }
        )

    def add_task(self, task_config, task_idx):
        t = {
            "idx": task_idx,
            "status": {
                "total_measurements": 0,
                "SUCCESS_measurements": 0,
                "INSTANTIATION_ERROR_measurements": 0,
                "COMPILE_ERROR_measurements": 0,
                "COMPILE_DEVICE_ERROR_measurements": 0,
                "RUNTIME_ERROR_measurement": 0,
                "WRONG_ANSWER_ERROR_measurements": 0,
                "BUILD_TIMEOUT_ERROR_measurements": 0,
                "RUN_TIMEOUT_ERROR_measurements": 0,
                "UNKNOWN_ERROR_measurements": 0
            }
        }
        t.update(task_config)
        self.exp["data"]["tasks"].append(t)
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.tasks": self.exp["data"]["tasks"]
            }
        )

    def add_method(self, method_spec, method_idx):
        m = {
            "idx": method_idx,
            "kind": method_spec["kind"],
            "schedulable": method_spec["scheduling"],
            "name": method_spec["name"],
            "cost_model": method_spec["cost_model"],
            "optimizer": method_spec["optimizer"],
            "search_policy": method_spec["search_policy"],
            "sampler": method_spec["sampler"],
            "filter": method_spec["filter"]
        }
        self.exp["data"]["methods"].append(m)
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.methods": self.exp["data"]["methods"]
            }
        )

    def update_task_status(self, idx, measurements, counts):
        for k in range(len(ERROR_TYPES)):
            key = ERROR_TYPES[k] + "_measurements"
            if key in self.exp["data"]["tasks"][idx]["status"]:
                self.exp["data"]["tasks"][idx]["status"][key] += counts[k]
        self.exp["data"]["tasks"][idx]["status"]["total_measurements"] += measurements
        self.s.update_one(
            "experiments",
            {"id": self.exp["id"]},
            {
                "data.tasks": self.exp["data"]["tasks"]
            }
        )


    def set_target(self, target):
        self.exp["data"]["target"] = target
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.target": self.exp["data"]["target"]
            }
        )

    def set_target_host(self, target_host):
        self.exp["data"]["target_host"] = target_host
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.target_host": self.exp["data"]["target_host"]
            }
        )

    def add_build_duration(self, dur):
        self.exp["data"]["build_duration"] += dur 
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.build_duration": self.exp["data"]["build_duration"]
            }
        )

    def add_run_duration(self, dur):
        self.exp["data"]["run_duration"] += dur 
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.run_duration": self.exp["data"]["run_duration"]
            }
        )
        
    def add_rank_duration(self, dur):
        self.exp["data"]["rank_duration"] += dur 
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.rank_duration": self.exp["data"]["rank_duration"]
            }
        )

    def __exit__(self, exc_type, exc_value, exc_traceback):
        Experiment.__exit__(self, exc_type, exc_value, exc_traceback)
        self.exp["data"]["calc_duration"] = self.exp["duration"] - (self.exp["data"]["build_duration"] + self.exp["data"]["rank_duration"] + self.exp["data"]["run_duration"])
        self.s.update_one(
            "experiments", 
            {"id": self.exp["id"]}, 
            {
                "data.calc_duration": self.exp["data"]["calc_duration"]
            }
        )

experiments = {
    "executor:model": ModelExecutorExperiment,
    "executor:tensor_program": TensorProgramExecutorExperiment,
    "compiler:model": ModelCompilerExperiment,
    "compiler:tensor_program": TensorProgramCompilerExperiment,
    "orchestrator:default": OrchestratorExperiment
}


