models = {
    "vgg16": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
# "vgg16-int8": {
#     "model": "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-12-int8.onnx",
#     "default_input_shape": (3, 224, 224),
#     "default_output_shape": (1000, ),
#     "dtype": "int8"
# },
    "vgg16_bn": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-bn-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg19": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg19-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg19_bn": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg19-bn-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg19_caffe2": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg19-caffe2-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "squeezenet1_0": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "squeezenet1_1": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet18_v1": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet34_v1": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet34-v1-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet50_v1": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    # "resnet50_v1_int8": {
    #     "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-12-int8.onnx",
    #     "default_input_shape": (3, 224, 224),
    #     "default_output_shape": (1000, ),
    #     "dtype": "int8"
    # },
    "resnet101_v1": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v1-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet152_v1": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v1-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet18_v2": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v2-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet34_v2": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet34-v2-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet50_v2": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet101_v2": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v2-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet152_v2": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v2-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v2_1_00": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "caffenet": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/caffenet/model/caffenet-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "googlenet": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "alexnet": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "shufflenet_v1_1_00": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/shufflenet/model/shufflenet-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "shufflenet_v2_1_00": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/shufflenet/model/shufflenet-v2-10.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
# "shufflenet_v2_1_00_int8": {
#     "model": "https://github.com/onnx/models/raw/master/vision/classification/shufflenet/model/shufflenet-v2-12-int8.onnx",
#     "default_input_shape": (3, 224, 224),
#     "default_output_shape": (1000, ),
#     "dtype": "int8"
# },
    "inception_v1": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "inception_v2": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "densenet121": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/densenet-121/model/densenet-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "rcnn": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "efficientnet_b4_lite": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "zfnet512": {
        "model": "https://github.com/onnx/models/raw/master/vision/classification/zfnet-512/model/zfnet512-9.onnx",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    }

}

import os

from tvm import relay
from conductor.utils import download
from conductor._base import get_conductor_path

DEFAULT_INPUT_NAMES = ["data_0", "data", "input", "gpu_0/data_0", "images:0"]

def acquire(model_name, details, custom_path=None, force_reload=False):
    import onnx
    """
    Acquires model preexisting details such as pre-trained weights or configuration
    Specific to the framework

    Parameters
    ----------
    model_name : str
        name of the model
    details : dict
        details of the model as registered in the framework definition module
    custom_path : str, optional
        whether to use a custom path for storage
    force_reload : bool, optional
        whether to forcefully reload (usually re-download) any preexisting model details

    Returns
    -------
    any
        framework specific acquisition result
    """

    models_pth = custom_path if custom_path is not None else os.path.join(get_conductor_path(), "models")
    model_pth = os.path.join(models_pth, model_name + "_" + "onnx" + ".cond.framework")
    os.makedirs(model_pth, exist_ok=True)
    if not os.path.exists(os.path.join(model_pth, model_name + ".onnx")) or force_reload:
        return download(details["model"], os.path.join(model_pth, model_name + ".onnx"))
    else:
        return os.path.join(model_pth, model_name + ".onnx")

def instantiate(model_name, details, acq_result, batch_size=1, in_shape=None, out_shape=None, dtype="float32"):
    """
    Instantiates model given parameters and any acquisition result
    The resultant instance is framework-specific

    Parameters
    ----------
    model_name : str
        name of the model
    details : dict
        details of the model as registered in the framework definition module
    acq_result : any
        framework-specific acquisition result as obtained from acquire()
    batch_size : int, optional
        batch size for the model
    in_shape: tuple, optional
        input shape of the model (tuple of ints)
    out_shape: tuple, optional
        output shape of the model (tuple of ints) i.e. (1000,)
    dtype : str, optional
        data type to be used for the model
    """

    model = onnx.load(acq_result)
    return model

def convert(model_instance, batch_size, input_shape, output_shape, dtype="float32"):
    """
    Converts framework-specific model instance object into TVM IR Module

    Parameters
    ----------
    model_instance : any
        framework-specific model instance
    batch_size : int
        batch size for the model
    input_shape: tuple
        input shape of the model (tuple of ints)
    output_shape: tuple
        output shape of the model (tuple of ints) i.e. (1000,)
    dtype : str, optional
        data type to be used for the model
    """
    
    input_all = model_instance.graph.input
    
    input_name = None
    for ia in input_all:
        for pin in DEFAULT_INPUT_NAMES:
            if ia.name == pin:
                input_name = ia.name
                break

    if input_name is not None:
        return (relay.frontend.from_onnx(model_instance, shape={input_name: (batch_size,) + input_shape}, dtype=dtype), {
            "input_name": input_name,
            "batch_size": batch_size,
            "input_shape": input_shape,
            "output_shape": output_shape
        })
    else:
        raise RuntimeError("Uknown input_name in onnx model")


