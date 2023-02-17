models = {
    "squeezenet1_0": {
        "model": "https://download.pytorch.org/models/squeezenet1_0-a815701f.pth",
        "load_name": "squeezenet1_0",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "squeezenet1_1": {
        "model": "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth",
        "load_name": "squeezenet1_1",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "mobilenet_v2_1_00": {
        'model': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        "load_name": "mobilenet_v2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "mobilenet_v3_1_00_large": {
        "model": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        "load_name": "mobilenet_v3_large",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "mobilenet_v3_1_00_small": {
        "model": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        "load_name": "mobilenet_v3_small",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "resnet152_v1": {
        'model': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        "load_name": "resnet152",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "resnet101_v1": {
        'model': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        "load_name": "resnet101",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "resnet50_v1": {
        'model': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        "load_name": "resnet50",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "resnet34_v1": {
        'model': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        "load_name": "resnet34",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "resnet18_v1": {
        "model": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "load_name": "resnet18",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "resnext50_32x4d": {
        'model': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        "load_name": "resnext50_32x4d",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "resnext101_32x8d": {
        'model': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        "load_name": "resnext101_32x8d",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "wide_resnet50_2": {
        'model': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        "load_name": "wide_resnet50_2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "wide_resnet101_2": {
        'model': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        "load_name": "wide_resnet101_2",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg19": {
        'model': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        "load_name": "vgg19",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg19_bn": {
        'model': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        "load_name": "vgg19_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg16": {
        'model': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        "load_name": "vgg16",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg16_bn": {
        'model': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        "load_name": "vgg16_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg13": {
        'model': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        "load_name": "vgg13",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg13_bn": {
        'model': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        "load_name": "vgg13_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg11": {
        'model': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        "load_name": "vgg11",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "vgg11_bn": {
        'model': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        "load_name": "vgg11_bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "densenet201": {
        'model': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        "load_name": "densenet201",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
        "expand": True
    },
    "densenet161": {
        'model': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
        "load_name": "densenet161",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
        "expand": True
    },
    "densenet169": {
        'model': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        "load_name": "densenet169",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
        "expand": True
    },
    "densenet121": {
        'model': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        "load_name": "densenet121",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
        "expand": True
    },
    "alexnet": {
        'model': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
        "load_name": "alexnet",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "shufflenet_v2_0_50": {
        'model': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
        "load_name": "shufflenet_v2_x0_5",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "shufflenet_v2_1_00": {
        'model': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
        "load_name": "shufflenet_v2_x1_0",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "mnasnet0_50": {
        "model": "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
        "load_name": "mnasnet0_5",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "mnasnet1_00": {
        "model": "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
        "load_name": "mnasnet1_0",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },

    "inception_v3": {
        "model": 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
        "load_name": "inception_v3",
        "default_input_shape": (3, 299, 299),
        "default_output_shape": (1000, ),
    },
    "googlenet": {
        'model': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
        "load_name": "googlenet",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    },
    "convnext_large": {
        "model": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        "load_name": "convnext_large",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000, ),
    }
}

import os
import re

from tvm import relay
from conductor._base import get_conductor_path

DEFAULT_INPUT_NAMES = ["main_input0"]


def acquire(model_name, details, custom_path=None, force_reload=False):
    import torch
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
    model_pth = os.path.join(models_pth, model_name + "_" + "pytorch" + ".cond.framework")
    os.makedirs(model_pth, exist_ok=True)
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model_path = os.path.join(model_pth, model_name + ".pth")
    dir_contents = os.listdir(model_pth)
    if len(dir_contents) < 1 or force_reload:
        scripted_model = torch.hub.load("pytorch/vision:v0.12.0", details["load_name"], pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(details["model"], model_dir=model_pth, map_location=None, progress=False, check_hash=False, file_name=model_name + ".pth")
        if "expand" in details:
            _dict_state = state_dict
            remove_data_parallel = False
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(_dict_state.keys()):
                match = pattern.match(key)
                new_key = match.group(1) + match.group(2) if match else key
                new_key = new_key[7:] if remove_data_parallel else new_key
                _dict_state[new_key] = _dict_state[key]
                # Delete old key only if modified.
                if match or remove_data_parallel: 
                    del _dict_state[key]
            scripted_model.load_state_dict(_dict_state)
        else:
            scripted_model.load_state_dict(state_dict)
    elif len(dir_contents) == 1:
        scripted_model = torch.hub.load("pytorch/vision:v0.12.0", details["load_name"], pretrained=False)
        state_dict = torch.load(model_path)
        if "expand" in details:
            _dict_state = state_dict
            remove_data_parallel = False
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(_dict_state.keys()):
                match = pattern.match(key)
                new_key = match.group(1) + match.group(2) if match else key
                new_key = new_key[7:] if remove_data_parallel else new_key
                _dict_state[new_key] = _dict_state[key]
                # Delete old key only if modified.
                if match or remove_data_parallel: 
                    del _dict_state[key]
            scripted_model.load_state_dict(_dict_state)
        else:
            scripted_model.load_state_dict(state_dict)
    return (scripted_model, model_path)

def instantiate(model_name, details, acq_result, batch_size=1, in_shape=None, out_shape=None, dtype="float32"):
    import torch
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

    if in_shape is not None:
        input_shape = in_shape
    else:
        input_shape = details["default_input_shape"]
    input_data = torch.randn(list((batch_size, ) + input_shape))
    model, _ = acq_result
    model = model.eval()
    scripted_model = torch.jit.trace(
        model,
        input_data,
        check_trace=False
    ).eval()
    return scripted_model

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
    
    model_obj = relay.frontend.from_pytorch(model_instance, [(DEFAULT_INPUT_NAMES[0], (batch_size, ) + input_shape)], default_dtype=dtype)
    return (model_obj, {
        "input_name": DEFAULT_INPUT_NAMES[0],
        "batch_size": batch_size,
        "input_shape": input_shape,
        "output_shape": output_shape
    })