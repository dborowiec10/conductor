import tvm
from tvm.auto_scheduler.measure import ProgramMeasurer

from conductor._base import Configurable
from conductor.component.measurer.nvidia_mps import NvidiaMPS
from conductor.experiment import Experiment

class Measurer(Configurable):
    _name = "measurer"
    
    def __repr__(self):
        return Measurer._name

    def __init__(self, subtype, builder, runner, configs=None, child_default_configs={}):
        self.builder = builder
        self.runner = runner
        self.configurer = None
        self.program_measurer = None
        self.hash_callback = None
        self.stage = None
        self.task_idx = 0
        self.remove_built_schedule = True
        
        Configurable.__init__(self, "measurer", [subtype], configs, Configurable.merge_configs({
            "target": "cuda",
            "target_host": "llvm",
            "devices": [],
            "nvidia_mps_devices": []
        }, child_default_configs, override_first=True))

        if len(self.config.get("nvidia_mps_devices", [])) > 0:
            self.using_mps = "mps"
            self.mps = NvidiaMPS(self.config.get("nvidia_mps_devices", []))
            self.mps.stop(None)
            self.mps_dir = self.mps.start()
        else:
            self.using_mps = "no_mps"
            self.mps = None
            self.mps_dir = None
        
        self.target = self.config["target"]
        self.target_host = self.config["target_host"]
        Experiment.current.set_target(self.target)
        Experiment.current.set_target_host(self.target_host)
        self.device_ids = self.config["devices"]
        self.num_devs = len(self.device_ids)

    def set_stage(self, stage):
        self.stage = stage

    def set_task_idx(self, task_idx):
        self.task_idx = task_idx

    def set_hash_callback(self, hash_callback):
        self.hash_callback = hash_callback

    def set_configurer(self, configurer):
        self.configurer = configurer
        self.program_measurer = ProgramMeasurer(self.builder.builder, self.runner.runner, [], 0)

    def set_orch_scheduler(self, scheduler):
        if self.builder is not None:
            self.builder.set_orch_scheduler(scheduler)
        if self.runner is not None:
            self.runner.set_orch_scheduler(scheduler)

    def load(self):
        tvm._ffi.register_func("auto_scheduler.measurer.measure", f=self.measure_wrapper, override=True)
        self.builder.load()
        self.runner.load(self.num_devs, devices=self.device_ids)
        self.runner.set_target(self.target)
        self.runner.set_target_host(self.target_host)
        
    def unload(self):
        self.runner.unload()
        self.builder.unload()

    def measure_wrapper(self, task, configs):
        _, mrs, _ = self.measure(task, configs, options=None)
        return mrs

    def measure(self, task, configs, options=None):
        raise NotImplementedError("abstract method, needs implementation in child class!")
