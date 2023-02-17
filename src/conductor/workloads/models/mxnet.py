

models = {
    "squeezenet1_0": {
        "load_name": "squeezenet1.0",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "squeezenet1_1": {
        "load_name": "squeezenet1.1",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v2_1_00": {
        "load_name": "mobilenetv2_1.0",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v2_0_75": {
        "load_name": "mobilenetv2_0.75",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v2_0_50": {
        "load_name": "mobilenetv2_0.5",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v2_0_25": {
        "load_name": "mobilenetv2_0.25",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00": {
        "load_name": "mobilenet1.0",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75": {
        "load_name": "mobilenet0.75",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50": {
        "load_name": "mobilenet0.5",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25": {
        "load_name": "mobilenet0.25",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet152_v2": {
        "load_name": "resnet152_v2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet101_v2": {
        "load_name": "resnet101_v2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet50_v2": {
        "load_name": "resnet50_v2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet34_v2": {
        "load_name": "resnet34_v2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet18_v2": {
        "load_name": "resnet18_v2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet152_v1": {
        "load_name": "resnet152_v1",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet101_v1": {
        "load_name": "resnet101_v1",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet50_v1": {
        "load_name": "resnet50_v1",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet34_v1": {
        "load_name": "resnet34_v1",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "resnet18_v1": {
        "load_name": "resnet18_v1",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg19": {
        "load_name": "vgg19",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg19_bn": {
        "load_name": "vgg19_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg16": {
        "load_name": "vgg16",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg16_bn": {
        "load_name": "vgg16_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg13": {
        "load_name": "vgg13",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg13_bn": {
        "load_name": "vgg13_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg11": {
        "load_name": "vgg11",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "vgg11_bn": {
        "load_name": "vgg11_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "densenet201": {
        "load_name": "densenet201",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "densenet161": {
        "load_name": "densenet161",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "densenet169": {
        "load_name": "densenet169",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    },
    "densenet121": {
        "load_name": "densenet121",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, )
    }
}

import os
from tvm import relay
from conductor._base import get_conductor_path

DEFAULT_INPUT_NAMES = ["main_input0"]

def acquire(model_name, details, custom_path=None, force_reload=False):
    import mxnet
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
    model_pth = os.path.join(models_pth, model_name + "_" + "mxnet" + ".cond.framework")
    os.makedirs(model_pth, exist_ok=True) 
    model = mxnet.gluon.model_zoo.vision.get_model(details["load_name"], pretrained=True, root=model_pth)
    model_params_path = None
    for fname in os.listdir(model_pth):
        if fname.startswith(details["load_name"]):
            model_params_path = os.path.join(model_pth, fname)
            os.rename(model_params_path, os.path.join(model_pth, model_name + ".params"))
            model_params_path = os.path.join(model_pth, model_name + ".params")
            break
    return (model, model_params_path)

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

    model, model_params_path = acq_result
    model.load_parameters(model_params_path, ctx=mxnet.cpu())
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
    
    model_obj = relay.frontend.from_mxnet(model_instance, {DEFAULT_INPUT_NAMES[0]: (batch_size,) + input_shape}, dtype=dtype)
    return (model_obj, {
        "input_name": DEFAULT_INPUT_NAMES[0],
        "batch_size": batch_size,
        "input_shape": input_shape,
        "output_shape": output_shape
    })