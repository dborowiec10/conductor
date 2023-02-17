
from _pytest.mark import param
import pytest
import os
from conductor.model import Model
from conductor.workloads.models.tvmmods import models, acquire, convert, instantiate
from conductor.workloads.models.pytorch import models as pytorch_models, acquire as pytorch_acquire, convert as pytorch_convert, instantiate as pytorch_instantiate

@pytest.mark.skip()
def test_model_instantiation():
    m = Model("mlp", "tvm")
    assert m is not None

@pytest.mark.skip()
def test_model_instantiation_from_framework_object_tvm():
    test_input = ("mlp", models["mlp"])
    acq_res = acquire(test_input[0], test_input[1])
    assert acq_res is None
    out = instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
    assert out is not None
    mod = Model.from_framework_object("mlp", "tvm", out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
    assert mod is not None

@pytest.mark.skip()
def test_model_instantiation_from_framework_object_pytorch():
    test_input = ("googlenet", pytorch_models["googlenet"])
    acq_res = pytorch_acquire(test_input[0], test_input[1])
    assert acq_res is not None
    out = pytorch_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=test_input[1]["default_input_shape"], out_shape=test_input[1]["default_output_shape"])
    assert out is not None
    mod = Model.from_framework_object("googlenet", "pytorch", out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
    assert mod is not None

@pytest.mark.skip()
def test_model_instantiation_from_json_params():
    m1 = Model("mlp", "tvm")
    m1.save_json()
    m1.save_params()
    assert m1 is not None
    json_path = os.path.join("/", "home", "user", ".conductor", "models", "mlp_tvm.cond", "graph.json")
    params_path = os.path.join("/", "home", "user", ".conductor", "models", "mlp_tvm.cond", "params.bin")
    assert os.path.exists(json_path)
    assert os.path.exists(params_path)
    m2 = Model.from_json_and_params("mlp", "tvm", json_path, params_path)
    assert m2 is not None

@pytest.mark.skip()
def test_model_save():
    m1 = Model("mlp", "tvm")
    assert m1 is not None
    m1.save()
    model_path = os.path.join("/", "home", "user", ".conductor", "models", "mlp_tvm.cond", "model.cond")
    assert os.path.exists(model_path)
    
@pytest.mark.skip()
def test_model_load():
    m1 = Model("mlp", "tvm")
    assert m1 is not None
    m1.save()
    model_path = os.path.join("/", "home", "user", ".conductor", "models", "mlp_tvm.cond", "model.cond")
    assert os.path.exists(model_path)
    m2 = Model.load(model_path)
    assert m2 is not None