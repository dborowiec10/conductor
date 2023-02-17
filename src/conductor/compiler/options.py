import os

pass_configs = {
    "relay.FuseOps.max_depth": int,
    "tir.detect_global_barrier": bool,
    "tir.LoopPartition": dict, # partition_const_loop: bool, no_unroll_loop_with_extent_one: bool
    "relay.fallback_device_type": int,
    "tir.HoistIfThenElse": dict, # support_block_scope_hosting: bool
    "tir.disable_vectorize": bool,
    "tir.noalias": bool,
    "tir.is_entry_func": bool,
    "tir.instrument_bound_checkers": bool,
    "tir.disable_assert": bool,
    "tir.debug_keep_trivial_loop": bool,
    "tir.InjectDoubleBuffer": dict, # split_loop: int
    "relay.backend.use_auto_scheduler": bool,
    "tir.add_lower_pass": list,
    "tir.UnrollLoop": dict #   auto_max_step: int, auto_max_depth: int, auto_max_extent: int, explicit_unroll: int
}

compilation_passes = {
    "LowerTensorExpr": 0,
    "AnnotateTargetFunc": 0,
    "PlanDevicesRewrite": 0,
    "FakeQuantizationToInteger": 0,
    "FoldExplicitPadding": 0,
    "FuseOps": 0,
    "MergeCompilerRegions": 0,
    "MergeComposite": 0,
    "SimplifyExpr": 0,
    "SimplifyInference": 0,
    "ToMixedPrecision": 0,
    "QuantizeAnnotate": 1,
    "QuantizePartition": 1,
    "QuantizeRealize": 1,
    "DeadCodeElimination": 1,
    "LabelOps": 1,
    "Legalize": 1,
    "SplitArgs": 1,
    "ToCPS": 1,
    "UnCPS": 1,
    "ToGraphNoramlForm": 1,
    "DynamicToStatic": 2,
    "FoldConstant": 2,
    "LazyGradientInit": 2,
    "AlterOpLayout": 3,
    "AutoSchedulerLayoutRewrite": 3,
    "CanonicalizeCast": 3,
    "CanonicalizeOps": 3,
    "ConvertLayout": 3,
    "DefuseOps": 3,
    "EliminateCommonSubexpr": 3,
    "ForwardFoldScaleAxis": 3,
    "BackwardFoldScaleAxis": 3,
    "CombineParallelBatchMatmul": 4,
    "CombineParallelConv2d": 4,
    "CombineParallelDense": 4,
    "CombineParallelOpBatch": 4,
    "Conv2dToSparse": 4,
    "DenseToSparse": 4,
    "FastMath": 4,
    "SimplifyFCTranspose": 4,
    "Conv2dToSparse2": 5
}

class TensorProgramCompilationOptions(object):
    _name = "tensor_program_compilation_options"

    def __repr__(self):
        return TensorProgramCompilationOptions._name

    @staticmethod
    def validate(_dict):
        if "save_tensor_program" in _dict:
            assert _dict["save_tensor_program"] in [True, False], "save_tensor_program must be <True, False> for tensor program compilation options"
        if "save_source" in _dict:
            assert _dict["save_source"] in [True, False], "save_source must be <True, False> for tensor program compilation options"
        if "instr_count" in _dict:
            assert _dict["instr_count"] in [True, False], "instr_count must be <True, False> for tensor program compilation options"
        if "ext_compiler" in _dict:
            assert isinstance(_dict["ext_compiler"], (str, type(None))), "ext_compiler must be str/None for tensor program compilation options"
        if "ext_compiler_options" in _dict:
            assert isinstance(_dict["ext_compiler_options"],  (str, type(None))), "ext_compiler_options must be str for tensor program compilation options"
        if "implementations_type" in _dict:
            assert _dict["implementations_type"] in ["template", "sketch", "flex", "tophub", None], "invalid implementations type specified from tensor program compilation options"
            if _dict["implementations_type"] in ["template", "sketch", "flex"]:
                assert "implementations_path" in _dict, "implementations_path not specified for implementations_type <template, sketch, flex> for tensor program compilation options"
                assert os.path.exists(_dict["implementations_path"])

        if "pass_configurations" in _dict:
            assert isinstance(_dict["pass_configurations"], dict), "pass_configurations must be a dict for tensor program compilation options"
            for k, d in _dict["pass_configurations"]:
                assert k in pass_configs, "unknown pass configuration for tensor program compilation options"
                assert isinstance(d, pass_configs[k]), "data type of pass configuration option does not match the option for tensor program compilation options"
        
        if "target" in _dict:
            assert isinstance(_dict["target"], str), "target must be str for tensor program compilation options"
    
        if "target_host" in _dict:
            assert isinstance(_dict["target_host"], str), "target_host must be str for tensor program compilation options"

        if "target_opts" in _dict:
            assert isinstance(_dict["target_opts"], (str, type(None))), "target_opts must be str/None for tensor program compilation options"
        
        if "custom_path" in _dict:
            assert isinstance(_dict["custom_path"], str)
            assert os.path.exists(_dict["custom_path"])

    @staticmethod
    def from_dict(_dict):
        return TensorProgramCompilationOptions(**_dict)

    def __init__(self,
        target="llvm",
        target_host="llvm",
        target_opts=None,
        save_tensor_program=False,
        save_source=False,
        ext_compiler=None,
        ext_compiler_options=None,
        instr_count=False,
        pass_configurations={},
        implementations_type=None,
        implementations_path=None,
        custom_path=None
    ):
        self.pass_configurations = pass_configurations
        self.save_tensor_program = save_tensor_program
        self.save_source = save_source
        self.instr_count = instr_count
        self.implementations_path = implementations_path
        self.implementations_type = implementations_type
        self.ext_compiler = ext_compiler
        self.ext_compiler_options = ext_compiler_options
        self.target = target
        self.target_host = target_host
        self.target_opts = target_opts
        self.custom_path = custom_path


class ModelCompilationOptions(object):
    _name = "model_compilation_options"

    def __repr__(self):
        return ModelCompilationOptions._name

    @staticmethod
    def validate(_dict):
        if "save_model" in _dict:
            assert _dict["save_model"] in [True, False], "save_model must be <True, False> for model compilation options"
        if "save_source" in _dict:
            assert _dict["save_source"] in [True, False], "save_source must be <True, False> for model compilation options"
        if "instr_count" in _dict:
            assert _dict["instr_count"] in [True, False], "instr_count must be <True, False> for model compilation options"
        if "ext_compiler" in _dict:
            assert isinstance(_dict["ext_compiler"], (str, type(None))), "ext_compiler must be str/None for model compilation options"
        if "ext_compiler_options" in _dict:
            assert isinstance(_dict["ext_compiler_options"], (str, type(None))), "ext_compiler_options must be str for model compilation options"
        if "implementations_type" in _dict:
            assert _dict["implementations_type"] in ["template", "sketch", "flex", "tophub", None], "invalid implementations type specified from model compilation options"
            if _dict["implementations_type"] in ["template", "sketch", "flex"]:
                assert "implementations_path" in _dict, "implementations_path not specified for implementations_type <template, sketch, flex> for model compilation options"
                assert os.path.exists(_dict["implementations_path"])
        if "layout" in _dict:
            assert isinstance(_dict["layout"], str), "layout must be str for model compilation options"
            assert _dict["layout"] in ["NCHW", "NHWC"]
        if "pass_configurations" in _dict:
            assert isinstance(_dict["pass_configurations"], dict), "pass_configurations must be a dict for model compilation options"
            for k, d in _dict["pass_configurations"]:
                assert k in pass_configs, "unknown pass configuration for model compilation options"
                assert isinstance(d, pass_configs[k]), "data type of pass configuration option does not match the option for model compilation options"
        
        _opt_level = 3

        if "opt_level" in _dict:
            assert isinstance(_dict["opt_level"], int), "opt_level must be int for model compilation options"
            assert _dict["opt_level"] in [0, 1, 2, 3, 4, 5], "opt_level must be in [0, 1, 2, 3, 4, 5] for model compilation options"
            _opt_level = _dict["opt_level"]

        if "target" in _dict:
            assert isinstance(_dict["target"], str), "target must be str for model compilation options"
    
        if "target_host" in _dict:
            assert isinstance(_dict["target_host"], str), "target_host must be str for model compilation options"

        if "target_opts" in _dict:
            assert isinstance(_dict["target_opts"], (str, type(None))), "target_opts must be str/None for model compilation options"

        if "required_passes" in _dict:
            assert isinstance(_dict["required_passes"], list), "required_passes must be a list for model compilation options"
            for i in _dict["required_passes"]:
                assert i in compilation_passes, "unknown required compilation pass for model compilation options"
                assert compilation_passes[i] <= _opt_level, "Required pass cannot be enabled with specified opt_level for model compilation options"
    
        if "disabled_passes" in _dict:
            assert isinstance(_dict["disabled_passes"], list), "disabled_passes must be a list for model compilation options"
            for i in _dict["disabled_passes"]:
                assert i in compilation_passes, "unknown disabled compilation pass for model compilation options"
                assert compilation_passes[i] <= _opt_level, "Disabled pass cannot be disabled with specified opt_level for model compilation options"

        if "custom_path" in _dict:
            assert isinstance(_dict["custom_path"], str)
            assert os.path.exists(_dict["custom_path"])

    @staticmethod
    def from_dict(_dict):
        ModelCompilationOptions.validate(_dict)
        return ModelCompilationOptions(**_dict)

    def __init__(self,
        target="llvm",
        target_host="llvm",
        target_opts=None,
        save_model=False,
        save_source=False,
        ext_compiler=None,
        ext_compiler_options=None,
        layout="NCHW",
        opt_level=3,
        instr_count=False,
        required_passes=[],
        disabled_passes=[],
        pass_configurations={},
        implementations_type=None,
        implementations_path=None,
        custom_path=None
    ):
        self.pass_configurations = pass_configurations
        self.save_model = save_model
        self.save_source = save_source
        self.instr_count = instr_count
        self.implementations_path = implementations_path
        self.implementations_type = implementations_type
        self.ext_compiler = ext_compiler
        self.ext_compiler_options = ext_compiler_options
        if layout is not None:
            self.layouts = {
                "nn.conv2d": [layout, "default"],
                "nn.conv2d_transpose": [layout, "default"],
                "qnn.conv2d": [layout, "default"],
            }
        else:
            self.layouts = None
        self.layout = layout
        self.opt_level = opt_level
        
        self.required_passes = required_passes
        self.disabled_passes = disabled_passes
        self.target = target
        self.target_host = target_host
        self.target_opts = target_opts
        self.custom_path = custom_path