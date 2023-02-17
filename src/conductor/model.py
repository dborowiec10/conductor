

import os
import logging
import cloudpickle
import tvm
import json
from conductor._base import get_conductor_path, InputOutputSpecification

from conductor.workloads.models.tvmmods import instantiate as tvm_instantiate, convert as tvm_convert, acquire as tvm_acquire, models as tvm_models, DEFAULT_INPUT_NAMES as tvm_default_inp_names
from conductor.workloads.models.keras import instantiate as keras_instantiate, convert as keras_convert, acquire as keras_acquire, models as keras_models, DEFAULT_INPUT_NAMES as keras_default_inp_names
from conductor.workloads.models.pytorch import instantiate as pytorch_instantiate, convert as pytorch_convert, acquire as pytorch_acquire, models as pytorch_models, DEFAULT_INPUT_NAMES as pytorch_default_inp_names
from conductor.workloads.models.mxnet import instantiate as mxnet_instantiate, convert as mxnet_convert, acquire as mxnet_acquire, models as mxnet_models, DEFAULT_INPUT_NAMES as mxnet_default_inp_names
from conductor.workloads.models.paddle import instantiate as paddle_instantiate, convert as paddle_convert, acquire as paddle_acquire, models as paddle_models, DEFAULT_INPUT_NAMES as paddle_default_inp_names
from conductor.workloads.models.tflite import instantiate as tflite_instantiate, convert as tflite_convert, acquire as tflite_acquire, models as tflite_models, DEFAULT_INPUT_NAMES as tflite_default_inp_names
from conductor.workloads.models.onnx import instantiate as onnx_instantiate, convert as onnx_convert, acquire as onnx_acquire, models as onnx_models, DEFAULT_INPUT_NAMES as onnx_default_inp_names

frameworks = {
    "tvm": {
        "models": tvm_models,
        "acquire": tvm_acquire,
        "convert": tvm_convert,
        "instantiate": tvm_instantiate,
        "default_inp_name": tvm_default_inp_names
    },
    "pytorch": {
        "models": pytorch_models,
        "acquire": pytorch_acquire,
        "convert": pytorch_convert,
        "instantiate": pytorch_instantiate,
        "default_inp_name": pytorch_default_inp_names
    },
    "mxnet": {
        "models": mxnet_models,
        "acquire": mxnet_acquire,
        "convert": mxnet_convert,
        "instantiate": mxnet_instantiate,
        "default_inp_name": mxnet_default_inp_names
    },
    "keras": {
        "models": keras_models,
        "acquire": keras_acquire,
        "convert": keras_convert,
        "instantiate": keras_instantiate,
        "default_inp_name": keras_default_inp_names
    },
    "paddle": {
        "models": paddle_models,
        "acquire": paddle_acquire,
        "convert": paddle_convert,
        "instantiate": paddle_instantiate,
        "default_inp_name": paddle_default_inp_names
    },
    "tflite": {
        "models": tflite_models,
        "acquire": tflite_acquire,
        "convert": tflite_convert,
        "instantiate": tflite_instantiate,
        "default_inp_name": tflite_default_inp_names
    },
    "onnx": {
        "models": onnx_models,
        "acquire": onnx_acquire,
        "convert": onnx_convert,
        "instantiate": onnx_instantiate,
        "default_inp_name": onnx_default_inp_names
    }
}

logger_model = logging.getLogger("conductor.model")
logger_compiled_model = logging.getLogger("conductor.compiled_model")

class Model(object):
    _name = "model"
    """
    Represents a DL model within the Conductor ecosystem

    Attributes
    ----------
    model_name : str
        model name as string
    model_framerwork : str
        framrwork name as string
    model_input : str
        hmmm....
    """

    def __repr__(self):
        return Model._name

    def __init__(self, model_name, model_framework, batch_size=1, in_shape=None, out_shape=None, custom_path=None, load=False, force_reload=False):
        """
        Creates a DL Model within the Conductor ecosystem
        Attributes
        ----------
        model_name : str
            model name as string
        model_framework : str
            framework name as string
        batch_size : int, optional
            batch size for the model
        in_shape : tuple, optional
            tuple of ints representing the input shape model override
        out_shape : tuple, optional
            tuple of ints representing the output shape model override
        custom_path : str, optional
            string representing custom path where to save the model files
            !Must be a directory!
            !If doesn't exist, will be created!
        force_reload : bool, optional
            whether to reload any framework-related model files (i.e. re-download weights from the web)
        
        Raises
        ------
        ValueError
            If model_name or model_framework not found within the conductor registry
        """
        self.model_name = model_name
        self.model_framework = model_framework
        self.model_input_spec = None
        self.model_json = None
        self.model_params = None
        self.model_batch_size = batch_size
        self.model_in_shape_override = in_shape
        self.model_out_shape_override = out_shape
        self.model_in_shape = None
        self.model_out_shape = None
        self.model_custom_path = custom_path
        self.force_reload = force_reload

        if self.model_name is not None and self.model_framework is not None:
            self._set_path(custom_path=custom_path)

            os.makedirs(self.model_path, exist_ok=True)

            if self.model_framework not in frameworks:
                raise ValueError("Unknown model framework")
            
            frm = frameworks[self.model_framework]
            
            if self.model_name not in frm["models"]:
                raise ValueError("Uknown model for framework: " + self.model_framework)

            model = frm["models"][self.model_name]

            self.model_in_shape = self.model_in_shape_override if self.model_in_shape_override is not None else model["default_input_shape"]
            self.model_out_shape = self.model_out_shape_override if self.model_out_shape_override is not None else model["default_output_shape"]

            if load:
                self._load()

    def _load(self):
        frm = frameworks[self.model_framework]
        acquirer = frm["acquire"]
        instantiator = frm["instantiate"]
        converter = frm["convert"]
        model = frm["models"][self.model_name]
        conv_inp = acquirer(self.model_name, model, custom_path=self.model_custom_path, force_reload=self.force_reload)

        inst = instantiator(
            self.model_name, 
            model,
            conv_inp,
            batch_size=self.model_batch_size,
            in_shape=self.model_in_shape,
            out_shape=self.model_out_shape
        )

        mod_obj, _dict = converter(
            inst, self.model_batch_size,
            self.model_in_shape,
            self.model_out_shape
        )

        mod_json, mod_params = mod_obj

        self.model_json = mod_json
        self.model_params = mod_params
        self.model_input_spec = _dict
        self.save_params()
        self.save_json()

    def _set_path(self, custom_path=None) -> None:
        # Sets model path
        # custom_path : str, optional
        #    custom path for the model to be used
        base_path = custom_path if custom_path is not None else os.path.join(get_conductor_path(), "models")
        self.model_path = os.path.join(base_path, self.model_name + "_" + self.model_framework + ".cond")

    def get_input(self, fill="random") -> None:
        """
        Parameters
        ----------
        fill: str or int or float, optional
            type of fill: <random>, <zeros>, <ones>, value|int, float|
        
        Returns
        -------

        Retrieves sample input for the model
        
        """
        pass

    def input_from_numpy(self, numpy_input):
        """
        Generates TVM input for the model from numpy array

        """
        pass

    def get_input_spec(self) -> dict:
        """
        Retrieves model input specification

        Returns
        -------
        dict
            A dictionary representing model input specification
        """
        return self.model_input_spec

    def get_json(self) -> dict:
        """
        Retrieves model graph json definition

        Returns
        -------
        dict
            A dictionary representing the json definition of model graph
        """
        return self.model_json

    def get_params(self) -> dict:
        """
        Retrieves model graph json definition

        Returns
        -------
        dict
            A dictionary representing the json definition of model graph
        """
        return self.model_params

    def save(self, path=None, force=False) -> None:
        """
        Serializes and saves the current model state

        Parameters
        ----------
        path : str, optional
            Custom path to use when saving the model
            ! The path will be augmented with suffix *.cond*
            If not specified, ~/.conductor/models/<model_name>_<model_framework>.cond/model.cond will be used as path
        force : bool, optional
            whether to force re-saving of the model
        """
        dst = os.path.join(
                self.model_path if path is None else path, 
                "model.cond"
            )
        if (not os.path.exists(dst)) or force:
            self.save_json()
            d = {
                "json": tvm.ir.save_json(self.model_json),
                "params": tvm.relay.save_param_dict(self.model_params),
                "input_spec": self.model_input_spec,
                "name": self.model_name,
                "framework": self.model_framework,
                "batch_size": self.model_batch_size,
                "in_shape": self.model_in_shape,
                "out_shape": self.model_out_shape,
                "custom_path": self.model_custom_path,
            }
            with open(dst, "wb") as f:
                cloudpickle.dump(d, f)

    def save_json(self, path=None, force=False) -> None:
        """
        Saves model graph json
        
        Parameters
        ----------
        path: str, optional
            Custom path to use when saving the model
            ! The path will be augmwented with suffix *cond.graph.json*
            If not specified, ~/.conductor/models/<model_name>_<model_framework>.cond/graph.json will be used as path
        force : bool, optional
            whether to force re-saving of the model graph json
        """
        dst = os.path.join(self.model_path if path is None else path, "graph.json")
        if (not os.path.exists(dst)) or force:
            with open(dst, "w") as mf:
                mf.write(tvm.ir.save_json(self.model_json))

    def save_params(self, path=None, force=False) -> None:
        """
        Serializes and saves model parameters
        
        Parameters
        ----------
        path: str, optional
            Custom path to use when saving the model
            ! The path will be augmwented with suffix *cond.params.bin*
            If not specified, ~/.conductor/models/<model_name>_<model_framework>.cond/params.bin will be used as path
        force : bool, optional
            whether to force re-saving of the model parameters
        """
        dst = os.path.join(self.model_path if path is None else path, "params.bin")
        if (not os.path.exists(dst)) or force:
            with open(dst, "wb") as pf:
                pf.write(tvm.relay.save_param_dict(self.model_params))

    def load_json(self, path) -> None:
        """
        Loads model json

        Parameters
        ----------
        path: str
            Path to the model json file
        """
        with open(path, "r") as f:
            self.model_json = tvm.ir.load_json(f.read())

    def load_params(self, path) -> None:
        """
        Loads model params

        Parameters
        ----------
        path: str
            Path to the model params
        """
        with open(path, "rb") as f:
            self.model_params = tvm.relay.load_param_dict(f.read())

    def load_json_direct(self, json) -> None:
        """
        Loads json directly from raw data

        Parameters
        ----------
        json: any
            raw data object representing the json of the model
        """
        self.model_json = tvm.ir.load_json(json)

    def load_params_direct(self, params) -> None:
        """
        Loads params directly from raw data

        Parameters
        ----------
        params: any
            raw data object representing the params of the model
        """
        self.model_params = tvm.relay.load_param_dict(params)

    @staticmethod
    def from_framework_object(model_name, model_framework, obj, batch_size, in_shape, out_shape, custom_path=None):
        """
        Constructs a model from framework specific model instance

        Parameters
        ----------
        model_name: str
            name of the model
        model_framework: str
            name of the model framework
        obj: any
            object representing framework-specific model instance
        batch_size: int
            batch size of the model
        in_shape: tuple
            input shape of the model (tuple of ints)
        out_shape: tuple
            output shape of the model (tuple of ints) i.e. (1000,)
        custom_path: str, optional
            custom path to be used for model persistence
        
        Returns
        -------
        Model
            instance of the model
        """
        frm = frameworks[model_framework]
        converter = frm["convert"]
        mod_obj, _dict = converter(
            obj, batch_size,
            in_shape,
            out_shape
        )
        mod = Model(None, None)
        mod.model_name = model_name
        mod.model_framework = model_framework
        mod._set_path(custom_path=custom_path)
        mod_json, mod_params = mod_obj
        mod.model_json = mod_json
        mod.model_params = mod_params
        mod.model_input_spec = _dict
        mod.batch_size = batch_size
        mod.model_in_shape = in_shape
        mod.model_out_shape = out_shape
        mod.model_custom_path = custom_path
        return mod

    @staticmethod
    def load(path):
        """
        Loads a previously created model from specified file

        Parameters
        ----------
        path : str
            path to the persisted model file
        
        Returns
        -------
        Model
            model instance
        """
        with open(os.path.join(path), "rb") as f:
            d = cloudpickle.load(f)
            mod = Model(None, None)
            mod.load_json_direct(d["json"])
            mod.load_params_direct(d["params"])
            mod.model_input_spec = d["input_spec"]
            mod.model_name = d["name"]
            mod.model_framework = d["framework"]
            mod._set_path()
            mod.batch_size = d["batch_size"]
            mod.model_in_shape = d["in_shape"]
            mod.model_out_shape = d["out_shape"]
            mod.model_custom_path = d["custom_path"]
            mod._set_path(custom_path=d["custom_path"])
        return mod

    @staticmethod
    def from_json_and_params(model_name, model_framework, json_path, params_path, custom_path=None):
        """
        Constructs a model from previously saved model json graph and parameter files

        Parameters
        ----------
        model_name: str
            name of the model
        model_framework: str
            name of the model framework
        json_path: str
            path to the model graph json file
        params_path: str
            path to the model parameters file
        custom_path: str, optional
            custom path to be used for model persistence
        
        Returns
        -------
        Model
            instance of the model
        """
        mod = Model(None, None) 

        if model_framework not in frameworks:
            raise ValueError("Unknown model framework")
        
        frm = frameworks[model_framework]
        
        if model_name not in frm["models"]:
            raise ValueError("Uknown model for framework: " + model_framework)

        possible_inp_names = frm["default_inp_name"]
        mod.model_name = model_name
        mod.model_framework = model_framework
        
        mod._set_path(custom_path=custom_path)

        mod.load_json(json_path)
        mod.load_params(params_path)

        func_params = mod.get_json()["main"].params
        comb_input = None
        input_name = None
        for p in func_params:
            if str(p.name_hint) in possible_inp_names:
                comb_input = tuple(p.type_annotation.shape)
                input_name = p.name_hint
                break

        mod.model_batch_size = comb_input[0]
        mod.model_in_shape = tuple(list(comb_input)[1:])
        mod.model_out_shape = tuple(list(mod.get_json()["main"].ret_type.shape)[1:])
        mod.model_input_spec = {
            "input_name": input_name,
            "batch_size": mod.model_batch_size,
            "input_shape": mod.model_in_shape,
            "output_shape": mod.model_out_shape
        }

        return mod

class ModelInputOutputSpecification(InputOutputSpecification):
    _name = "model_input_output_specification"

    def __repr__(self):
        return InputOutputSpecification.__repr__(self) + ":" + ModelInputOutputSpecification._name

    @staticmethod
    def _validate(_dict):
        assert "model_name" in _dict, "model_name not in model_input specification"
        assert "model_framework" in _dict, "model_framerwork not in model_input specification"
        assert "batch_size" in _dict, "batch_size not in model_input specification"
        assert isinstance(_dict["batch_size"], int), "batch size is not int for model_input specification"
        if "in_shape" in _dict:
            assert isinstance(_dict["in_shape"], (list, tuple))
        if "out_shape" in _dict:
            assert isinstance(_dict["out_shape"], (list, tuple))
        if "force_reload" in _dict:
            assert _dict["force_reload"] in [False, True]
    
    def __init__(self, _dict):
        ModelInputOutputSpecification._validate(_dict)
        InputOutputSpecification.__init__(self, _dict)
        self.model_name = _dict["model_name"]
        self.model_framework = _dict["model_framework"]
        self.batch_size = _dict["batch_size"]
        self.in_shape = _dict.get("in_shape", None)
        self.out_shape = _dict.get("out_shape", None)
        self.force_reload = _dict.get("force_reload", False)

    def from_spec(self, models_path):
        return Model(
            self.model_name, self.model_framework, 
            batch_size=self.batch_size, in_shape=self.in_shape,
            out_shape=self.out_shape, custom_path=models_path,
            force_reload=self.force_reload
        )

class CompiledModel(object):
    _name = "compiled_model"

    def __repr__(self):
        return CompiledModel._name

    def __init__(
        self,
        model_name,
        model_framework,
        batch_size,
        opt_level,
        instr_count,
        implementations_type,
        implementations_path,
        in_shape,
        out_shape,
        mod_lib,
        mod_json,
        mod_params,
        custom_path=None,
        load=False,
        save_source=False,
        save_model=False
    ):
        self.model_name = model_name
        self.model_framework = model_framework
        self.model_batch_size = batch_size
        self.opt_level = opt_level
        self.instr_count = instr_count
        self.model_in_shape = in_shape
        self.model_out_shape = out_shape
        self.model_lib = mod_lib
        self.model_json = mod_json
        self.model_params = mod_params
        self.implementations_type = implementations_type
        self.implementations_path = implementations_path
        self.model_custom_path = custom_path
        self.model_path = None
        self.sources = {
            "cu": None,
            "relay": None,
            "s": None
        }

        if in_shape is not None and out_shape is not None and mod_json is not None and mod_params is not None and mod_lib is not None:
            self._set_path(custom_path=custom_path)
            self.model_in_shape = in_shape
            self.model_out_shape = out_shape
            self.model_lib = mod_lib
            self.model_json = mod_json
            self.model_params = mod_params
            
            if save_model:
                self.save_json(self.model_json)
                self.save_params(self.model_params)
                self.save_lib(self.model_lib)

            if save_source:
                self.save_source(self.model_lib)

        else:
            self._set_path(custom_path=custom_path)
            if load:
                self._load()

    def get_lib(self):
        if self.model_lib is not None:
            return self.model_lib
        else:
            self.model_lib = self.load_lib()
            return self.model_lib

    def get_json(self):
        if self.model_json is not None:
            return self.model_json
        else:
            self.model_json = self.load_json()
            return self.model_json
    
    def get_params(self):
        if self.model_params is not None:
            return self.model_params
        else:
            self.model_params = self.load_params()
            return self.model_params
    
    def save_json(self, _json):
        with open(os.path.join(self.model_path, "graph.json"), "w") as _file:
            json.dump(_json, _file)

    def save_params(self, params):
        with open(os.path.join(self.model_path, "params.bin"), "wb") as _file:
            _file.write(tvm.relay.save_param_dict(params))
    
    def save_lib(self, lib):
        lib.export_library(os.path.join(self.model_path, "lib.so"))

    def save_source(self, lib):
        source_types = ["s", "relay", "cu"]
        for s in source_types:
            with open(os.path.join(self.model_path, "source." + s), "w") as src_file:
                if s == "relay":
                    src_file.write(lib.get_source())
                elif s == "cu":
                    src_file.write(lib.imported_modules[0].get_source("cu"))
                else:
                    src_file.write(lib.get_source(s))

    def load_json(self):
        with open(os.path.join(self.model_path, "graph.json"), "r") as _file:
            return json.load(_file)

    def load_params(self):
        with open(os.path.join(self.model_path, "params.bin"), "rb") as _file:
            return tvm.relay.load_param_dict(_file.read())

    def load_lib(self):
        return tvm.runtime.load_module(os.path.join(self.model_path, "lib.so"))

    def load_sources(self):
        for k, _ in self.sources.items():
            with open(os.path.join(self.model_path, "source." + k), "r") as _file:
                self.sources[k] = _file.read()

    def get_source(self, _format="cu"):
        assert _format in self.sources, "invalid source format requested"
        if self.sources[_format] is None:
            self.load_sources()
        return self.sources[_format]

    # used for when CompiledModel is cloudpickled across processes
    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self.model_framework = state["model_framework"]
        self.model_batch_size = state["model_batch_size"]
        self.opt_level = state["opt_level"]
        self.instr_count = state["instr_count"]
        self.implementations_type = state["implementations_type"]
        self.implementations_path = state["implementations_path"]
        self.model_custom_path = state["model_custom_path"]
        self.model_path = state["model_path"]
        self.sources = state["sources"]
        self.model_in_shape = state["model_in_shape"]
        self.model_out_shape = state["model_out_shape"]
        self.model_lib = None
        self.model_json = None
        self.model_params = None

    def __getstate__(self):
        return {
            "model_name": self.model_name,
            "model_framework": self.model_framework,
            "model_batch_size": self.model_batch_size,
            "opt_level": self.opt_level,
            "instr_count": self.instr_count,
            "implementations_type": self.implementations_type,
            "implementations_path": self.implementations_path,
            "model_custom_path": self.model_custom_path,
            "model_path": self.model_path,
            "sources": self.sources,
            "model_in_shape": self.model_in_shape,
            "model_out_shape": self.model_out_shape
        }

    def _load(self):
        with open(os.path.join(self.model_path, "model.compiled.cond"), "rb") as f:
            d = cloudpickle.load(f)
            self.model_in_shape = d["in_shape"]
            self.model_out_shape = d["out_shape"]
            self.model_json = self.load_json()
            self.model_params = self.load_params()
            self.model_lib = self.load_lib()
            self.load_sources()

    # load from directory
    @staticmethod
    def load(path):
        with open(os.path.join(path, "model.compiled.cond"), "rb") as f:
            d = cloudpickle.load(f)
            mod = CompiledModel(
                d["name"],
                d["framework"],
                d["batch_size"],
                d["opt_level"],
                d["instr_count"],
                d["implementations_type"],
                d["implementations_path"],
                None, None, None, None, None,
                custom_path=d["custom_path"]
            )
            mod.model_in_shape = d["in_shape"]
            mod.model_out_shape = d["out_shape"]
            mod.model_json = mod.load_json()
            mod.model_params = mod.load_params()
            mod.model_lib = mod.load_lib()
            mod.load_sources()
        return mod

    def save(self, path=None) -> None:
        self.save_json(self.model_json)
        self.save_params(self.model_params)
        self.save_lib(self.model_lib)
        self.save_source(self.model_lib)
        dst = os.path.join(self.model_path if path is None else path, "model.compiled.cond")
        d = {
            "name": self.model_name,
            "framework": self.model_framework,
            "batch_size": self.model_batch_size,
            "opt_level": self.opt_level,
            "instr_count": self.instr_count,
            "implementations_type": self.implementations_type,
            "implementations_path": self.implementations_path,
            "in_shape": self.model_in_shape,
            "out_shape": self.model_out_shape,
            "custom_path": self.model_custom_path
        }
        with open(dst, "wb") as f:
            cloudpickle.dump(d, f)

    def _set_path(self, custom_path=None) -> None:
        base_path = custom_path if custom_path is not None else os.path.join(get_conductor_path(), "models")
        _mod_name = self.model_name + "_" + self.model_framework
        _mod_name += "_b" + str(self.model_batch_size)
        _mod_name += "_o" + str(self.opt_level)
        if self.instr_count == True:
            _mod_name += "_ic"
        _mod_name += ".compiled.cond"
        self.model_path = os.path.join(base_path, _mod_name)
        os.makedirs(self.model_path, exist_ok=True)

class CompiledModelInputOutputSpecification(InputOutputSpecification):
    _name = "compiled_model_input_output_specification"
    
    def __repr__(self):
        return InputOutputSpecification.__repr__(self) + ":" + CompiledModelInputOutputSpecification._name

    @staticmethod
    def _validate(_dict):
        assert "model_name" in _dict, "model_name not in compiled_model_input specification"
        assert "model_framework" in _dict, "model_framerwork not in compiled_model_input specification"
        assert "batch_size" in _dict, "batch_size not in compiled_model_input specification"
        assert "opt_level" in _dict, "opt_level not in compiled_model_input specification"
        assert _dict["opt_level"] in [0, 1, 2, 3, 4, 5], "incorrect opt_level for compiled_model_input specification"
        if "instr_count" in _dict:
            assert _dict["instr_count"] in [True, False], "instr count must be True/False for compiled_model_input specification"
        if "implementations_type" in _dict:
            assert isinstance(_dict["implementations"], (str, type(None))), "implementations_type must be str/None for compiled_model_input_specification"
            if _dict["implementations_type"] not in [None, "tophub"]:
                assert _dict["implementations_type"] in ["sketch", "flex", "template"]
                assert "implementations_path" in _dict
                assert isinstance(_dict["implementations_path"], str)
    
    def __init__(self, _dict):
        CompiledModelInputOutputSpecification._validate(_dict)
        InputOutputSpecification.__init__(self, _dict)
        self.model_name = _dict["model_name"]
        self.model_framework = _dict["model_framework"]
        self.batch_size = _dict["batch_size"]
        self.opt_level = _dict["opt_level"]
        self.instr_count = _dict.get("instr_count", False)
        self.implementations_type = _dict.get("implementations_type", None)
        self.implementations_path = _dict.get("implementations_path", None)
        
    def from_spec(self, models_path):
        return CompiledModel(
            self.model_name,
            self.model_framework,
            self.batch_size,
            self.opt_level,
            self.instr_count,
            self.implementations_type,
            self.implementations_path,
            None, None, None, None, None,
            load=True,
            custom_path=models_path
        )

