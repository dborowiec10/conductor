from conductor._base import Specification
from conductor.model import CompiledModel
from conductor.tensor_program import CompiledTensorProgram
from conductor.executor.options import ExecutionOptions
from conductor.component.evaluator.evaluator import evaluators

class ExecutionSpecification(Specification):
    _name = "execution_specification"
    
    def __repr__(self):
        return Specification.__repr__(self) + ":" + ExecutionSpecification._name

    def _validate(self, _dict):
        assert isinstance(_dict, dict)
        assert "name" in _dict, "name not specified for execution specification"
        assert "type" in _dict, "type not specified for execution specification"
        assert _dict["type"] == "execution"
        assert "spec" in _dict, "spec not specified for execution specification"
        assert "group_type" in _dict["spec"], "group_type not specified for execution specification"
        assert _dict["spec"]["group_type"] in ["tensor_program", "model"], "unknown group_type for execution specification: valid <tensor_program, model>"
        assert "evaluator" in _dict["spec"], "evaluator not specified for execution specification"
        assert isinstance(_dict["spec"]["evaluator"], str), "evaluator must be str for execution specification"
        assert _dict["spec"]["evaluator"] in evaluators, "unknown evaluator for execution specification"
        
        assert "options" in _dict["spec"], "options not specified for execution specification"
        ExecutionOptions.validate(_dict["spec"]["options"])

        assert "maps" in _dict["spec"], "maps not specified for execution specification"
        assert isinstance(_dict["spec"]["maps"], list), "maps is not a list for execution specification"
        for m in _dict["spec"]["maps"]:
            assert isinstance(m, dict), "each map in execution specification must be dict"
            assert "input" in m, "input not specified for execution specification map"
            assert isinstance(m["input"], str), "input is not a string for execution specification map"

    def pair_inputs_options(self, inputs, configs):
        pairs = []
        _evaluator = self.evaluator(configs=configs)
        if self.group_type == "model":
            for m in self.maps:
                assert m["input"] in inputs,  "input specified in execution specification map is not present in the global inputs for the task"
                assert isinstance(inputs[m["input"]], CompiledModel), "invalid input type for execution specification map with group_type=model"
                pairs.append((inputs[m["input"]], self.options, _evaluator))

        elif self.group_type == "tensor_program":
            for m in self.maps:
                assert m["input"] in inputs,  "input specified in execution specification map is not present in the global inputs for the task"
                assert isinstance(inputs[m["input"]], CompiledTensorProgram), "invalid input type for execution specification map with group_type=tensor_program"
                pairs.append((inputs[m["input"]], self.options, _evaluator))
        return pairs

    def __init__(self, _dict):
        self._validate(_dict)
        Specification.__init__(self, _dict["name"], _dict["type"])
        self.evaluator = evaluators[_dict["spec"]["evaluator"]]
        self.group_type = _dict["spec"]["group_type"]
        self.options = ExecutionOptions.from_dict(_dict["spec"]["options"])
        self.maps = _dict["spec"]["maps"]