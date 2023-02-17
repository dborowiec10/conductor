
import tvm

models = {
    "mlp": {
        "load_name": "mlp",
        "default_input_shape": (1, 28, 28),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "dqn": { # TODO: needs some attention in terms of model input
        "load_name": "dqn",
        "default_input_shape": (4, 84, 84),
        "default_output_shape": (18,)
    },
    "dcgan": {
        "load_name": "dcgan",
        "default_input_shape": (100,),
        "default_output_shape": (3, 64, 64),
        "kwargs": {
            "oshape": (3, 64, 64)
        }
    },
    "squeezenet1_0": {
        "load_name": "squeezenet",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "version": "1.0", 
            "num_classes": 1000
        }
    },
    "squeezenet1_1": {
        "load_name": "squeezenet",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "version": "1.1",
            "num_classes": 1000
        }
    },
    "inception_v3": {
        "load_name": "inception_v3",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "mobilenet_v1_1_00": {
        "load_name": "mobilenet",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "resnet152_v1": {
        "load_name": "resnet-152",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "resnet101_v1": {
        "load_name": "resnet-101",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "resnet50_v1": {
        "load_name": "resnet-50",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "resnet34_v1": {
        "load_name": "resnet-34",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "resnet18_v1": {
        "load_name": "resnet-18",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "resnet_3d_50": {
        "load_name": "resnet_3d",
        "default_input_shape": (3, 16, 112, 112),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_layers": 50,
            "num_classes": 1000
        }
    },
    "resnet_3d_34": {
        "load_name": "resnet_3d",
        "default_input_shape": (3, 16, 112, 112),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_layers": 34,
            "num_classes": 1000
        }
    },
    "resnet_3d_18": {
        "load_name": "resnet_3d",
        "default_output_shape": (1000,),
        "default_input_shape": (3, 16, 112, 112),
        "kwargs": {
            "num_layers": 18,
            "num_classes": 1000
        }
    },
    "vgg19": {
        "load_name": "vgg-19",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "vgg19_bn": {
        "load_name": "vgg-19-bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "vgg16": {
        "load_name": "vgg-16",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "vgg16_bn": {
        "load_name": "vgg-16-bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "vgg13": {
        "load_name": "vgg-13",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "vgg13_bn": {
        "load_name": "vgg-13-bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "vgg11": {
        "load_name": "vgg-11",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    },
    "vgg11_bn": {
        "load_name": "vgg-11-bn",
        "default_input_shape": (3, 224, 224),
        "default_output_shape": (1000,),
        "kwargs": {
            "num_classes": 1000
        }
    }
}

DEFAULT_INPUT_NAMES = ["data"]


def acquire(model_name, details, custom_path=None, force_reload=False):
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
    return None


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

    splits = details["load_name"].split("-")
    kwargs = details.get("kwargs", {})

    if in_shape is not None:
        input_shape = in_shape
    else:
        input_shape = details["default_input_shape"]

    if out_shape is not None:
        output_shape = out_shape
    else:
        output_shape = details["default_output_shape"]

    name = splits[0]

    _kwargs = {
        "batch_size": batch_size,
        "dtype": dtype,
    }
    if "oshape" in kwargs:
        _kwargs["oshape"] = output_shape
    else:
        _kwargs["image_shape"] = input_shape

    if len(splits) > 2 and splits[2] == "bn":
        _kwargs["batch_norm"] = True

    if len(splits) >= 2:
        _kwargs["num_layers"] = int(splits[1])
        
    _kwargs.update(kwargs)

    mod = tvm
    mod = getattr(mod, "relay")
    mod = getattr(mod, "testing")
    mod = getattr(mod, name)
    model_obj = mod.get_workload(**_kwargs)
    return model_obj

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
    
    return (model_instance, {
        "input_name": DEFAULT_INPUT_NAMES[0],
        "batch_size": batch_size,
        "input_shape": input_shape,
        "output_shape": output_shape
    })







