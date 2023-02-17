import cloudpickle
import os
import tvm
from conductor._base import get_conductor_path, InputOutputSpecification
from conductor.utils import generic_topi_unwrapper
from tvm.auto_scheduler.compute_dag import ComputeDAG
from tvm.autotvm.task.task import compute_flop
from tvm.te import create_schedule


class TensorProgram(object):
    _name = "tensor_program"

    def __repr__(self):
        return TensorProgram._name

    def __init__(self, identifier, func_name, compute_gen_func, comp_sched_gen_func, args, target, target_host, load=False, custom_path=None, io_tensors=None, is_topi=False):
        self.identifier = identifier
        self.func_name = func_name
        self._set_path(custom_path=custom_path)
        self.flop = None
        if func_name is None and \
            compute_gen_func is None and \
            comp_sched_gen_func is None and \
            args is None and \
            target is None and \
            target_host is None:
            self.target = None
            self.target_host = None
            self.args = None
            self.compute_gen_func = None
            self.comp_sched_gen_func = None
            self.io_tensors = None
            self.is_topi = False
            if load:
                self._load()
            
        else:
            self.target = target
            self.target_host = target_host
            self.func_name = func_name
            self.args = args
            self.compute_gen_func = None
            self.comp_sched_gen_func = None
            self.io_tensors = None
            self.is_topi = is_topi
            
            if comp_sched_gen_func:
                self.comp_sched_gen_func = comp_sched_gen_func
                setattr(self.comp_sched_gen_func, "func_name", self.func_name)
                self.comp_sched_gen_func.__name__ = self.func_name
                self.comp_sched_gen_func.__qualname__ = self.func_name
            if compute_gen_func:
                
                if isinstance(compute_gen_func, list):
                    self.io_tensors = compute_gen_func
                    def cwrapper(*args, **kwargs):
                        return compute_gen_func
                    self.compute_gen_func = cwrapper
                    # self.compute_gen_func = compute_gen_func
                    setattr(self.compute_gen_func, "func_name", self.func_name)
                    self.compute_gen_func.__name__ = self.func_name
                    self.compute_gen_func.__qualname__ = self.func_name

                elif callable(compute_gen_func):
                    self.compute_gen_func = compute_gen_func
                    setattr(self.compute_gen_func, "func_name", self.func_name)
                    self.compute_gen_func.__name__ = self.func_name
                    self.compute_gen_func.__qualname__ = self.func_name
                else:
                    raise RuntimeError("unrecognisable definition for compute_gen_func")
                
                if io_tensors is not None:
                    self.io_tensors = io_tensors
                
            elif not comp_sched_gen_func:
                raise RuntimeError("Either: <compute_gen_func + comp_sched_gen_func>, <compute_gen_func> or <comp_sched_gen_func> must be specified")

            self.flop = self.get_flop()
    
    def is_templateable(self):
        if self.comp_sched_gen_func is None and self.compute_gen_func is None:
            # try and preload
            with open(os.path.join(self.tensor_program_path, "tensor_program.bin"), "rb") as pckl_file:
                retval = cloudpickle.load(pckl_file)
                assert "compute_gen_func" in retval
                assert "comp_sched_gen_func" in retval
                assert "io_tensors" in retval
                return True if retval["comp_sched_gen_func"] is not None else False
        else:
            return True if self.comp_sched_gen_func else False

    def __getstate__(self):
        return {
            "identifier": self.identifier,
            "func_name": self.func_name,
            "target": self.target,
            "target_host": self.target_host,
            "flop": self.flop,
            "compute_gen_func": self.compute_gen_func,
            "comp_sched_gen_func": self.comp_sched_gen_func,
            "io_tensors": self.io_tensors,
            "args": self.args,
            "is_topi": self.is_topi,
            "tensor_program_path": self.tensor_program_path
        }
        
    def get_flop(self):
        if self.flop == None:
            if self.is_topi:
                r = generic_topi_unwrapper(self.compute_gen_func, self.target, self.identifier)(*self.args)
                
                try:
                    s = create_schedule([x.op for x in [r]])
                    self.flop = compute_flop(s)
                except Exception:
                    # couldn't use autotvm (i.e. multi-op)
                    dag = ComputeDAG([r])
                    self.flop = dag.get_flop()
            else:
                dag = ComputeDAG(self.compute_gen_func(*self.args))
                self.flop = dag.get_flop()
        return self.flop

    def __setstate__(self, state):
        self.identifier = state["identifier"]
        self.func_name = state["func_name"]
        self.target = state["target"]
        self.target_host = state["target_host"]
        self.compute_gen_func = state["compute_gen_func"]
        self.comp_sched_gen_func = state["comp_sched_gen_func"]
        self.io_tensors = state["io_tensors"]
        self.args = state["args"]
        self.is_topi = state["is_topi"]
        self.flop = state["flop"]
        self.tensor_program_path = state["tensor_program_path"]

    def _set_path(self, custom_path=None) -> None:
        # Sets tensor_program path
        # custom_path : str, optional
        #    custom path for the tensor_program to be used
        base_path = custom_path if custom_path is not None else os.path.join(get_conductor_path(), "tensor_programs")
        self.tensor_program_path = os.path.join(base_path, self.identifier + ".cond")

    def save(self, path=None, force=False):
        dst = self.tensor_program_path if path is None else path
        os.makedirs(dst, exist_ok=True)
        dst = os.path.join(dst, "tensor_program.bin")
        if (not os.path.exists(dst)) or force:
            with open(dst, "wb") as pckl_file:
                cloudpickle.dump(self.__getstate__(), pckl_file)

    def _load(self):
        with open(os.path.join(self.tensor_program_path, "tensor_program.bin"), "rb") as pckl_file:
            retval = cloudpickle.load(pckl_file)
        self.__setstate__(retval)

    @staticmethod
    def load(path):
        with open(os.path.join(path, "tensor_program.bin"), "rb") as pckl_file:
            retval = cloudpickle.load(pckl_file)
        return TensorProgram(
            retval["identifier"],
            retval["func_name"],
            retval["target"],
            retval["target_host"],
            retval["compute_gen_func"],
            retval["comp_sched_gen_func"],
            retval["args"],
            custom_path=retval["tensor_program_path"],
            is_topi=retval["is_topi"],
            io_tensors=retval["io_tensors"]
        )

class TensorProgramInputOutputSpecification(InputOutputSpecification):
    _name = "tensor_program_input_output_specification"

    def __repr__(self):
        return InputOutputSpecification.__repr__(self) + ":" + TensorProgramInputOutputSpecification._name

    def __init__(self, _dict):
        InputOutputSpecification.__init__(self, _dict)
        self.identifier = _dict.get("identifier", None)

    def from_spec(self, tensor_programs_path):
        return TensorProgram(self.identifier, None, None, None, None, None, None, custom_path=tensor_programs_path)

class CompiledTensorProgram(object):
    _name = "compiled_tensor_program"

    def __repr__(self):
        return CompiledTensorProgram._name

    def __init__(
        self,
        identifier,
        args,
        func_name,
        instr_count,
        is_topi,
        implementations_type,
        implementations_path,
        tp_lib,
        custom_path=None,
        load=False,
        save_source=False,
        save_tensor_program=False
    ):
        self.identifier = identifier
        self.instr_count = instr_count
        self.is_topi = is_topi
        self.implementations_type = implementations_type
        self.implementations_path = implementations_path
        self.args = args
        self.func_name = func_name
        self.lib = tp_lib
        self.tensor_program_custom_path = custom_path
        self.tensor_program_path = None
        self.sources = {
            "cu": None,
            "relay": None,
            "s": None
        }

        if tp_lib is not None and args is not None and is_topi is not None and func_name is not None:
            self._set_path(custom_path=custom_path)
            self.lib = tp_lib
            self.args = args
            self.func_name = func_name
            
            if save_tensor_program:
                self.save_lib(tp_lib)

            if save_source:
                self.save_source(tp_lib)
        else:
            self._set_path(custom_path=custom_path)
            if load:
                self._load()

    def get_lib(self):
        if self.lib is not None:
            return self.lib
        else:
            self.lib = self.load_lib()
            return self.lib

    def save_lib(self, lib):
        lib.export_library(os.path.join(self.tensor_program_path, "lib.so"))

    def save_source(self, lib):
        source_types = ["s", "relay", "cu"]
        for s in source_types:
            with open(os.path.join(self.tensor_program_path, "source." + s), "w") as src_file:
                if s == "relay":
                    src_file.write(lib.get_source())
                elif s == "cu":
                    src_file.write(lib.imported_modules[0].get_source("cu"))
                else:
                    src_file.write(lib.get_source(s))

    def load_lib(self):
        return tvm.runtime.load_module(os.path.join(self.tensor_program_path, "lib.so"))

    def load_sources(self):
        for k, _ in self.sources.items():
            with open(os.path.join(self.tensor_program_path, "source." + k), "r") as _file:
                self.sources[k] = _file.read()

    def get_source(self, _format="cu"):
        assert _format in self.sources, "invalid source format requested"
        if self.sources[_format] is None:
            self.load_sources()
        return self.sources[_format]

    # used for when CompiledTensorProgram is cloudpickled across processes
    def __setstate__(self, state):
        self.identifier = state["identifier"]
        self.args = state["args"]
        self.func_name = state["func_name"]
        self.instr_count = state["instr_count"]
        self.is_topi = state["is_topi"]
        self.implementations_type = state["implementations_type"]
        self.implementations_path = state["implementations_path"]
        self.tensor_program_custom_path = state["tensor_program_custom_path"]
        self.tensor_program_path = state["tensor_program_path"]
        self.sources = state["sources"]
        self.lib = None

    def __getstate__(self):
        return {
            "identifier": self.identifier,
            "args": self.args,
            "func_name": self.func_name,
            "instr_count": self.instr_count,
            "is_topi": self.is_topi,
            "implementations_type": self.implementations_type,
            "implementations_path": self.implementations_path,
            "tensor_program_custom_path": self.tensor_program_custom_path,
            "tensor_program_path": self.tensor_program_path,
            "sources": self.sources
        }

    def _load(self):
        with open(os.path.join(self.tensor_program_path, "tensor_program.compiled.cond"), "rb") as f:
            d = cloudpickle.load(f)
            self.identifier = d["identifier"]
            self.args = d["args"]
            self.func_name = d["func_name"]
            self.instr_count = d["instr_count"]
            self.is_topi = d["is_topi"]
            self.implementations_type = d["implementations_type"]
            self.implementations_path = d["implementations_path"]
            self.lib = self.load_lib()

    # load from directory
    @staticmethod
    def load(path):
        with open(os.path.join(path, "tensor_program.compiled.cond"), "rb") as f:
            d = cloudpickle.load(f)
            tp = CompiledTensorProgram(
                d["identifier"],
                d["args"],
                d["func_name"],
                d["instr_count"],
                d["is_topi"],
                d["implementations_type"],
                d["implementations_path"],
                None, custom_path=d["custom_path"]
            )
            tp.lib = tp.load_lib()
            tp.load_sources()
            return tp

    def save(self, path=None) -> None:
        self.save_lib(self.lib)
        self.save_source(self.lib)
        dst = os.path.join(self.tensor_program_path if path is None else path, "tensor_program.compiled.cond")
        d = {
            "identifier": self.identifier,
            "args": self.args,
            "func_name": self.func_name,
            "instr_count": self.instr_count,
            "is_topi": self.is_topi,
            "implementations_type": self.implementations_type,
            "implementations_path": self.implementations_path,
            "custom_path": self.tensor_program_custom_path
        }
        with open(dst, "wb") as f:
            cloudpickle.dump(d, f)

    def _set_path(self, custom_path=None) -> None:
        base_path = custom_path if custom_path is not None else os.path.join(get_conductor_path(), "tensor_programs")
        _tp_name = self.identifier
        if self.instr_count == True:
            _tp_name += "_ic"
        if self.is_topi == True:
            _tp_name += "_topi"
        _tp_name += ".compiled.cond"

        self.tensor_program_path = os.path.join(base_path, _tp_name)
        os.makedirs(self.tensor_program_path, exist_ok=True)

class CompiledTensorProgramInputOutputSpecification(InputOutputSpecification):
    _name = "compiled_tensor_program_input_output_specification"
    
    def __repr__(self):
        return InputOutputSpecification.__repr__(self) + ":" + CompiledTensorProgramInputOutputSpecification._name

    @staticmethod
    def _validate(_dict):
        assert "identifier" in _dict, "identifier not in compiled_tensor_program specification"
        if "instr_count" in _dict:
            assert _dict["instr_count"] in [True, False], "instr_count must be True/False for compiled_tensor_program specification"
        if "is_topi" in _dict:
            assert _dict["is_topi"] in [True, False], "is_topi must be True/False for compiled_tensor_program specification"
        if "implementations_type" in _dict:
            assert isinstance(_dict["implementations"], (str, type(None))), "implementations_type must be str/None for compiled_tensor_program"
            if _dict["implementations_type"] not in [None, "tophub"]:
                assert _dict["implementations_type"] in ["sketch", "flex", "template"]
                assert "implementations_path" in _dict
                assert isinstance(_dict["implementations_path"], str)
    
    def __init__(self, _dict):
        CompiledTensorProgramInputOutputSpecification._validate(_dict)
        InputOutputSpecification.__init__(self, _dict)
        self.identifier = _dict["identifier"]
        self.instr_count = _dict.get("instr_count", False)
        self.is_topi = _dict.get("is_topi", False)
        self.implementations_type = _dict.get("implementations_type", None)
        self.implementations_path = _dict.get("implementations_path", None)
        
    def from_spec(self, tensor_programs_path):
        return CompiledTensorProgram(
            self.identifier,
            None,
            None,
            self.instr_count,
            self.is_topi,
            self.implementations_type,
            self.implementations_path,
            None,
            load=True,
            custom_path=tensor_programs_path
        )