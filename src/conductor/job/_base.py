import os

from conductor.job.job import Job
from conductor.job.tasks import tasks

from conductor._base import Specification, Configuration
from conductor.executor._base import ExecutionSpecification
from conductor.compiler._base import CompilationSpecification
from conductor.orchestrator._base import OrchestratorSpecification
from conductor.component.specification import MethodSpecification, InputMethodMapSpecification
from conductor.tensor_program import TensorProgramInputOutputSpecification, CompiledTensorProgramInputOutputSpecification
from conductor.model import ModelInputOutputSpecification, CompiledModelInputOutputSpecification
from conductor.profiler._base import ProfilerSpecification

specifications = {
    "compilation": CompilationSpecification,
    "execution": ExecutionSpecification,
    "orchestrator": OrchestratorSpecification,
    "method": MethodSpecification,
    "input_method_map": InputMethodMapSpecification,
    "tensor_program": TensorProgramInputOutputSpecification,
    "model": ModelInputOutputSpecification,
    "compiled_tensor_program": CompiledTensorProgramInputOutputSpecification,
    "compiled_model": CompiledModelInputOutputSpecification
}

class TaskSpecification(Specification):
    _name = "task_specification"

    def __repr__(self):
        return Specification.__repr__(self) + ":" + TaskSpecification._name
    
    def __init__(self, _dict, _configurations, _specifications):
        Specification.__init__(self, _dict["name"], _dict["type"])
        self.inputs = _dict["inputs"]
        self.outputs = _dict["outputs"]

        self.configurations = {}
        for c in _configurations:
            if c.name in _dict["configurations"]:
                self.configurations[c.name] = c

        self.specifications = {}
        for s in _specifications:
            if s.name in _dict["specifications"]:
                self.specifications[s.name] = s

        self.profilers = []
        for p in _dict["profilers"]:
            self.profilers.append(ProfilerSpecification(p, self.configurations))

    def from_spec(self, job_results_path, models_path, tensor_programs_path):
        klass = tasks[self._type]
        return klass(self, job_results_path, models_path, tensor_programs_path)

class JobSpecification(Specification):
    _name = "job_specification"

    def __repr__(self):
        return Specification.__repr__(self) + ":" + JobSpecification._name

    def __init__(self, _dict):
        Specification.__init__(self, _dict["job_name_prefix"], "job")
        self.inputs = [specifications[i["type"]](i) for i in _dict["inputs"]]
        self.specifications = [specifications[i["type"]](i) for i in _dict["specifications"]]
        self.meta = _dict.get("metadata", {})
        self.configurations = [Configuration(i) for i in _dict["configurations"]]
        self.task = TaskSpecification(_dict["task"], self.configurations, self.specifications)
        
    def from_spec(self, job_results_path, models_path, tensor_programs_path, results_db_spec):
        return Job(self, job_results_path, models_path, tensor_programs_path, results_db_spec)



