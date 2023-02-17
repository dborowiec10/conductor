from conductor._base import Specification
from conductor.utils import match_definition, check_definition
from conductor.model import Model
from conductor.tensor_program import TensorProgram
from conductor.component.method.method import methods
from conductor.component.cost_model.cost_model import cost_models
from conductor.component.optimizer.optimizer import optimizers
from conductor.component.search_policy.search_policy import search_policies
from conductor.component.sampler.sampler import samplers
from conductor.component.filter.filter import filters
from conductor.component.strategy.strategy import strategies

class InputMethodMapSpecification(Specification):
    _name = "input_method_map_specification"

    def __repr__(self):
        return Specification.__repr__(self) + ":" + InputMethodMapSpecification._name

    def _validate(self, _dict):
        assert isinstance(_dict, dict)
        assert "name" in _dict, "name not specified for input_method_map specification"
        assert "type" in _dict, "type not specified for input_method_map specification"
        assert _dict["type"] == "input_method_map"
        assert "spec" in _dict, "spec not specified for input_method_map specification"
        assert "strategy" in _dict["spec"], "strategy not specified for input_method_map specification"
        assert check_definition(strategies, _dict["spec"]["strategy"]), "unknown strategy for input_method_map specification"
        assert "group_type" in _dict["spec"], "group_type not specified for input_method_map specification"
        assert _dict["spec"]["group_type"] in ["tensor_program", "model"], "unknown group_type for input method map specification: valid <tensor_program, model>"
        assert "maps" in _dict["spec"], "maps not specified for input_method_map specification"
        assert isinstance(_dict["spec"]["maps"], list), "maps is not a list for input_method_map specification"
        for m in _dict["spec"]["maps"]:
            assert "input" in m, "input not specified for input_method_map specification map"
            assert isinstance(m["input"], str), "input is not a string for input_method_map specification map"
            assert "output" in m, "output not specified for input_method_map specification map"
            assert isinstance(m["output"], str), "output is not a string for input_method_map specification map"
            assert "method" in m, "method not specified for input_method_map specification map"
            assert isinstance(m["method"], str), "method is not a string for input_method_map specification map"

    def pair_inputs_methods(self, inputs, method_specs):
        model_maps = []
        tensor_program_maps = []
        if self.group_type == "model":
            for m in self.maps:
                assert m["input"] in inputs, "input specified in input_method_map map is not present in the global inputs for the task"
                assert isinstance(inputs[m["input"]], Model), "invalid input type for input_method_map with group_type=model"
                assert m["method"] in [mm.name for mm in method_specs], "method specified in input_method_map map is not present in the global list of method specifications for the task"
                model_maps.append({
                    "type": "model", 
                    "strategy_spec": self.strategy, 
                    "strategy": match_definition(strategies, self.strategy), 
                    "pairs": [
                        (
                            "model", 
                            m["input"], 
                            inputs[m["input"]], 
                            {mm.name: mm for mm in method_specs}[m["method"]]
                        )
                    ]
                })

        elif self.group_type == "tensor_program":
            tp_pairs = []
            for m in self.maps:
                assert m["input"] in inputs, "input specified in input_method_map map is not present in the global inputs for the task"
                assert isinstance(inputs[m["input"]], TensorProgram), "invalid input type for input_method_map with group_type=tensor_program"
                assert m["method"] in [mm.name for mm in method_specs], "method specified in input_method_map map is not present in the global list of method specifications for the task"
                for mm in method_specs:
                    if mm.name == m["method"]:
                        if mm.scheduling == "template":
                            assert inputs[m["input"]].is_templateable(), "input paired with template-based method in input_method_map specification is not templateable"
                tp_pairs.append((
                    "tensor_program", 
                    m["input"], 
                    inputs[m["input"]], 
                    {mm.name: mm for mm in method_specs}[m["method"]]
                ))
            
            tensor_program_maps.append({
                "type": "tensor_program", 
                "strategy_spec": self.strategy, 
                "strategy": match_definition(strategies, self.strategy), 
                "pairs": tp_pairs
            })

        return model_maps, tensor_program_maps

    def __init__(self, _dict):
        self._validate(_dict)
        Specification.__init__(self, _dict["name"], _dict["type"])
        self.strategy = _dict["spec"]["strategy"]
        self.group_type = _dict["spec"]["group_type"]
        self.maps = _dict["spec"]["maps"]

class MethodSpecification(Specification):
    _name = "method_specification"
    
    def __repr__(self):
        return Specification.__repr__(self) + ":" + MethodSpecification._name

    def _validate(self, _dict):
        assert "name" in _dict, "name not specified for method specification"
        assert "type" in _dict, "type not specified for method specification"
        assert _dict["type"] == "method"
        assert "spec" in _dict, "spec not specified for method specification"

        assert "scheduling" in _dict["spec"], "scheduling not specified for method specification"
        assert _dict["spec"]["scheduling"] is not None, "scheduling is none for method specification"
        assert check_definition(methods, _dict["spec"]["scheduling"]), "unknown scheduling for method specification"
       
        assert "kind" in _dict["spec"], "kind not specified for method specification"
        assert _dict["spec"]["kind"] in ["composite", "standalone"], "kind must be <composite, standalone> for method specification"

        if _dict["spec"]["kind"] == "standalone":
            assert "method_name" in _dict["spec"], "method_name not specified for method specification"
            assert check_definition(methods, _dict["spec"]["scheduling"] + ":standalone:" + _dict["spec"]["method_name"]), "unknown method name for method specification"
        
        elif _dict["spec"]["kind"] == "composite":            
            assert "method_name" not in _dict["spec"] or _dict["spec"]["method_name"] is None, "method name specified for composite method in method specification"
            assert check_definition(methods, _dict["spec"]["scheduling"] + ":composite")

            assert "cost_model" in _dict["spec"], "cost model not specified for method specification"
            assert check_definition(cost_models, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["cost_model"]), "unknown cost model specified in method specification"
         
            if _dict["spec"]["scheduling"] == "template":
                assert "optimizer" in _dict["spec"], "optimizer not specified for method specification"
                assert check_definition(optimizers, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["optimizer"]), "unknown optimizer for method specification"

                if "sampler" in _dict["spec"]:
                    if _dict["spec"]["sampler"] is not None:
                        assert check_definition(samplers, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["sampler"]), "unknown sampler for method specification"
                    
                if "filter" in _dict["spec"]:
                    if _dict["spec"]["filter"] is not None:
                        assert check_definition(filters, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["filter"]), "unknown filter for method specification"

            elif _dict["spec"]["scheduling"] == "sketch":
                assert "search_policy" in _dict["spec"]
                assert check_definition(search_policies, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["search_policy"]), "unknown search_policy for method specification"

    def __init__(self, _dict):
        self._validate(_dict)
        Specification.__init__(self, _dict["name"], _dict["type"])
        self.scheduling = _dict["spec"].get("scheduling", None)
        self.kind = _dict["spec"].get("kind", None)
        
        
        if self.kind == "composite":
            self.method_name = None
            self.cost_model_spec = _dict["spec"].get("cost_model", None)
            self.cost_model = match_definition(cost_models, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["cost_model"])
            self.method = match_definition(methods, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["kind"])

            if self.scheduling == "template":
                self.search_policy_spec = None
                self.search_policy = None
                self.optimizer_spec = _dict["spec"].get("optimizer", None)
                self.optimizer = match_definition(optimizers, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["optimizer"])
                self.sampler_spec = _dict["spec"].get("sampler", None)
                self.sampler = match_definition(samplers, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["sampler"]) if self.sampler_spec != None else None
                self.filter_spec = _dict["spec"].get("filter", None)
                self.filter = match_definition(filters, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["filter"]) if self.filter_spec != None else None

            elif self.scheduling == "sketch":
                self.search_policy_spec = _dict["spec"].get("search_policy", None)
                self.search_policy = match_definition(search_policies, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["search_policy"])
                self.optimizer_spec = None
                self.optimizer = None
                self.sampler_spec = None
                self.sampler = None
                self.filter_spec = None
                self.filter = None


        elif self.kind == "standalone":
            self.method_name = _dict["spec"].get("method_name", None)
            self.method = match_definition(methods, _dict["spec"]["scheduling"] + ":" + _dict["spec"]["kind"] + ":" + self.method_name)
            self.cost_model_spec = None
            self.cost_model = None
            self.search_policy_spec = None
            self.search_policy = None
            self.optimizer_spec = None
            self.optimizer = None
            self.sampler_spec = None
            self.sampler = None
            self.filter_spec = None
            self.filter = None
            