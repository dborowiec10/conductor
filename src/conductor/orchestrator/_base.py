from conductor._base import Configurable, Specification
from conductor.job.task import Task
from conductor.utils import match_definition, check_definition
from conductor.component.evaluator.evaluator import evaluators
from conductor.component.measurer.measurer import measurers
from conductor.component.builder.builder import builders
from conductor.component.runner.runner import runners

import logging
logger = logging.getLogger("conductor.component")

class Orchestrator(Task, Configurable):
    _name = "orchestrator"

    def __repr__(self):
        return Task.__repr__(self) + ":" + Orchestrator._name

    def __init__(self, subtype, spec, input_types, output_types, specification_types, task_results_path, models_path, tensor_programs_path, child_default_configs={}):
        Task.__init__(
            self, 
            subtype,
            spec,
            input_types,
            output_types,
            specification_types,
            task_results_path,
            models_path,
            tensor_programs_path
        )
        Configurable.__init__(self, "orchestrator", [subtype], self.spec.configurations, Configurable.merge_configs({}, child_default_configs, override_first=True))
        self.orch_spec = self.find_specifications_by_type("orchestrator")[0]

    def run(self, inputs, idx):
        raise NotImplementedError("Must override run")


class OrchestratorSpecification(Specification):
    _name = "OrchestratorSpecification"

    def __repr__(self):
        return Specification.__repr__(self) + ":" + OrchestratorSpecification._name

    def _validate(self, _dict):
        assert isinstance(_dict, dict)
        assert "name" in _dict, "name not specified for orchestrator specification"
        assert "type" in _dict, "type not specified for orchestrator specification"
        assert _dict["type"] == "orchestrator"
        assert "spec" in _dict, "spec not in orchestrator specification"
        assert "measurer" in _dict["spec"], "measurer not specified for orchestrator specification"
        assert check_definition(measurers, _dict["spec"]["measurer"]), "unknown measurer for orchestrator specification"
        assert "builder" in _dict["spec"], "builder not specified for orchestrator specification"
        assert check_definition(builders, _dict["spec"]["builder"]), "unknown builder for orchestrator specification"
        assert "runner" in _dict["spec"], "runner not specified for orchestrator specification"
        assert check_definition(runners, _dict["spec"]["runner"]), "unknown runner for orchestrator specification"
        assert "evaluator" in _dict["spec"], "evaluator not specified for orchestrator specification"
        assert check_definition(evaluators, _dict["spec"]["evaluator"]), "unknown evaluator for orchestrator specification"     
        assert "settings" in _dict, "settings not specified for orchestrator specification"
        assert _dict["settings"] is not None, "settings is none for orchestrator specification"
        assert isinstance(_dict["settings"], dict), "settings is not a dictionary for orchestrator specification"

    def __init__(self, _dict):
        self._validate(_dict)
        Specification.__init__(self, _dict["name"], _dict["type"])
        self.settings = _dict["settings"]
        self.measurer = match_definition(measurers, _dict["spec"]["measurer"])
        self.measurer_spec = _dict["spec"]["measurer"]
        self.evaluator = match_definition(evaluators, _dict["spec"]["evaluator"])
        self.evaluator_spec = _dict["spec"]["evaluator"]
        self.builder = match_definition(builders, _dict["spec"]["builder"])
        self.builder_spec = _dict["spec"]["builder"]
        self.runner = match_definition(runners, _dict["spec"]["runner"])
        self.runner_spec = _dict["spec"]["runner"]

