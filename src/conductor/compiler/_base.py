from conductor._base import Specification
from conductor.model import Model
from conductor.tensor_program import TensorProgram
from conductor.compiler.options import TensorProgramCompilationOptions, ModelCompilationOptions


class CompilationSpecification(Specification):
    _name = "compilation_specification"

    def __repr__(self):
        return Specification.__repr__(self) + ":" + CompilationSpecification._name

    def _validate(self, _dict):
        assert isinstance(_dict, dict)
        assert "name" in _dict, "name not specified for orchestrator specification"
        assert "type" in _dict, "type not specified for orchestrator specification"
        assert _dict["type"] == "compilation"
        assert "spec" in _dict, "spec not specified for compilation specification"
        assert "group_type" in _dict["spec"], "group_type not specified for compilation specification"
        assert _dict["spec"]["group_type"] in ["tensor_program", "model"], "unknown group_type for compilation specification: valid <tensor_program, model>"
        assert "options" in _dict["spec"], "options not specified for compilation specification"
        if _dict["spec"]["group_type"] == "model":
            ModelCompilationOptions.validate(_dict["spec"]["options"])
        else:
            TensorProgramCompilationOptions.validate(_dict["spec"]["options"])
        assert "maps" in _dict["spec"], "maps not specified for compilation specification"
        assert isinstance(_dict["spec"]["maps"], list), "maps is not a list for compilation specification"
        for m in _dict["spec"]["maps"]:
            assert isinstance(m, dict), "each map in compilation specification must be dict"
            assert "input" in m, "input not specified for compilation specification map"
            assert isinstance(m["input"], str), "input is not a string for compilation specification map"
            assert "output" in m, "output not specified for compilation specification map"
            assert isinstance(m["output"], str), "output is not a string for compilation specification map"


    def pair_inputs_options(self, inputs):
        pairs = []
        if self.group_type == "model":
            for m in self.maps:
                assert m["input"] in inputs,  "input specified in compilation specification map is not present in the global inputs for the task"
                assert isinstance(inputs[m["input"]], Model), "invalid input type for compilation specification map with group_type=model"
                pairs.append((inputs[m["input"]], self.options))

        elif self.group_type == "tensor_program":
            for m in self.maps:
                assert m["input"] in inputs,  "input specified in compilation specification map is not present in the global inputs for the task"
                assert isinstance(inputs[m["input"]], TensorProgram), "invalid input type for compilation specification map with group_type=tensor_program"
                assert self.options.implementations_type in [None, "tophub", "template", "sketch", "flex"], "invalid implementations_type option for tensor program input in compilation specification map"
                if self.options.implementations_type in ["tophub", "template"]:
                    assert inputs[m["input"]].is_templateable(), "tensor program input must be templateable for implementations_type -> <template, tophub>"
                pairs.append((inputs[m["input"]], self.options))
        return pairs

    def __init__(self, _dict):
        self._validate(_dict)
        Specification.__init__(self, _dict["name"], _dict["type"])
        self.group_type = _dict["spec"]["group_type"]
        if self.group_type == "model":
            self.options = ModelCompilationOptions.from_dict(_dict["spec"]["options"])
        elif self.group_type == "tensor_program":
            self.options = TensorProgramCompilationOptions.from_dict(_dict["spec"]["options"])
        self.maps = _dict["spec"]["maps"]
