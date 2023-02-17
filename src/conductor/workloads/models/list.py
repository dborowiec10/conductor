
from conductor.workloads.models.tvmmods import models as tvm_models
from conductor.workloads.models.keras import models as keras_models
from conductor.workloads.models.pytorch import  models as pytorch_models
from conductor.workloads.models.mxnet import models as mxnet_models
from conductor.workloads.models.paddle import models as paddle_models
from conductor.workloads.models.tflite import  models as tflite_models
from conductor.workloads.models.onnx import models as onnx_models

import tabulate

models = {}

for tvmmod, _ in tvm_models.items():
    if tvmmod not in models:
        models[tvmmod] = ["tvm"]
    else:
        models[tvmmod].append("tvm")

for kermod, _ in keras_models.items():
    if kermod not in models:
        models[kermod] = ["keras"]
    else:
        models[kermod].append("keras")

for pytmod, _ in pytorch_models.items():
    if pytmod not in models:
        models[pytmod] = ["pytorch"]
    else:
        models[pytmod].append("pytorch")

for mxnmod, _ in mxnet_models.items():
    if mxnmod not in models:
        models[mxnmod] = ["mxnet"]
    else:
        models[mxnmod].append("mxnet")

for padmod, _ in paddle_models.items():
    if padmod not in models:
        models[padmod] = ["paddle"]
    else:
        models[padmod].append("paddle")

for tflmod, _ in tflite_models.items():
    if tflmod not in models:
        models[tflmod] = ["tflite"]
    else:
        models[tflmod].append("tflite")

for onnxmod, _ in onnx_models.items():
    if onnxmod not in models:
        models[onnxmod] = ["onnx"]
    else:
        models[onnxmod].append("onnx")

frameworks = ["tvm", "keras", "pytorch", "mxnet", "paddle", "tflite", "onnx"]
data = []

for m, frams in models.items():
    dp = [m]
    for f in frameworks:
        if f in frams:
            dp.append("*")
        else:
            dp.append(" ")
    data.append(dp)

data = sorted(data, key=lambda x: x[0])

print(tabulate.tabulate(data, ["model"] + frameworks))

