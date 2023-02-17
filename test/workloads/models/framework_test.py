
import pytest

from conductor.framework.tvmmods import models as tvm_models
from conductor.framework.keras import models as keras_models
from conductor.framework.pytorch import  models as pytorch_models
from conductor.framework.mxnet import models as mxnet_models
from conductor.framework.paddle import models as paddle_models
from conductor.framework.tflite import  models as tflite_models
from conductor.framework.onnx import models as onnx_models

from conductor.framework.tvmmods import acquire as tvm_acquire
from conductor.framework.keras import acquire as keras_acquire
from conductor.framework.pytorch import  acquire as pytorch_acquire
from conductor.framework.mxnet import acquire as mxnet_acquire
from conductor.framework.paddle import acquire as paddle_acquire
from conductor.framework.tflite import  acquire as tflite_acquire
from conductor.framework.onnx import acquire as onnx_acquire

from conductor.framework.tvmmods import convert as tvm_convert
from conductor.framework.keras import convert as keras_convert
from conductor.framework.pytorch import  convert as pytorch_convert
from conductor.framework.mxnet import convert as mxnet_convert
from conductor.framework.paddle import convert as paddle_convert
from conductor.framework.tflite import  convert as tflite_convert
from conductor.framework.onnx import convert as onnx_convert

from conductor.framework.tvmmods import instantiate as tvm_instantiate
from conductor.framework.keras import instantiate as keras_instantiate
from conductor.framework.pytorch import  instantiate as pytorch_instantiate
from conductor.framework.mxnet import instantiate as mxnet_instantiate
from conductor.framework.paddle import instantiate as paddle_instantiate
from conductor.framework.tflite import  instantiate as tflite_instantiate
from conductor.framework.onnx import instantiate as onnx_instantiate


how_many = 2

# @pytest.mark.skip()
# @pytest.mark.parametrize("test_input", [(k, det) for (k, det) in list(keras_models.items())][:how_many])
# def test_keras_acquire(test_input):
#     acq_res = keras_acquire(test_input[0], test_input[1])
#     assert acq_res is not None
#     out = keras_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
#     assert out is not None
#     mod_obj, _dict = keras_convert(out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
#     assert mod_obj is not None
#     assert _dict is not None

# @pytest.mark.skip()
# @pytest.mark.parametrize("test_input", [(k, det) for (k, det) in pytorch_models.items()][:how_many])
# def test_pytorch_acquire(test_input):
#     acq_res = pytorch_acquire(test_input[0], test_input[1])
#     assert acq_res is not None
#     out = pytorch_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
#     assert out is not None
#     mod_obj, _dict = pytorch_convert(out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
#     assert mod_obj is not None
#     assert _dict is not None

# @pytest.mark.skip()
# @pytest.mark.parametrize("test_input", [(k, det) for (k, det) in tvm_models.items()][:how_many])
# def test_tvm_acquire(test_input):
#     acq_res = tvm_acquire(test_input[0], test_input[1])
#     assert acq_res is None
#     out = tvm_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
#     assert out is not None
#     mod_obj, _dict = tvm_convert(out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
#     assert mod_obj is not None
#     assert _dict is not None

# @pytest.mark.skip()
# @pytest.mark.parametrize("test_input", [(k, det) for (k, det) in mxnet_models.items()][:how_many])
# def test_mxnet_acquire(test_input):
#     acq_res = mxnet_acquire(test_input[0], test_input[1])
#     assert acq_res is not None
#     out = mxnet_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
#     assert out is not None
#     mod_obj, _dict = mxnet_convert(out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
#     assert mod_obj is not None
#     assert _dict is not None

# @pytest.mark.skip()
# @pytest.mark.parametrize("test_input", [(k, det) for (k, det) in onnx_models.items()][:how_many])
# def test_onnx_acquire(test_input):
#     acq_res = onnx_acquire(test_input[0], test_input[1])
#     assert acq_res is not None
#     out = onnx_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
#     assert out is not None
#     mod_obj, _dict = onnx_convert(out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
#     assert mod_obj is not None
#     assert _dict is not None

# @pytest.mark.skip()
# @pytest.mark.parametrize("test_input", [(k, det) for (k, det) in tflite_models.items()][:how_many])
# def test_tflite_acquire(test_input):
#     acq_res = tflite_acquire(test_input[0], test_input[1])
#     assert acq_res is not None
#     out = tflite_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
#     assert out is not None
#     mod_obj, _dict = tflite_convert(out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
#     assert mod_obj is not None
#     assert _dict is not None

# @pytest.mark.skip()
# @pytest.mark.parametrize("test_input", [(k, det) for (k, det) in paddle_models.items()][:how_many])
# def test_paddle_acquire(test_input):
#     acq_res = paddle_acquire(test_input[0], test_input[1])
#     assert acq_res is not None
#     out = paddle_instantiate(test_input[0], test_input[1], acq_res, batch_size=1, in_shape=None, out_shape=None)
#     assert out is not None
#     mod_obj, _dict = paddle_convert(out, 1, test_input[1]["default_input_shape"], test_input[1]["default_output_shape"])
#     assert mod_obj is not None
#     assert _dict is not None



