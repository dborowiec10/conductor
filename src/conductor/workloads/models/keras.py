

models = {
    "xception": {
        "load_name": "xception.Xception",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (299, 299, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "inception_v3": {
        "load_name": "inception_v3.InceptionV3",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (299, 299, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "mobilenet_v2_1_00": {
        "load_name": "mobilenet_v2.MobileNetV2",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 1}
    },
    "mobilenet_v2_0_75": {
        "load_name": "mobilenet_v2.MobileNetV2",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 0.75}
    },
    "mobilenet_v2_0_50": {
        "load_name": "mobilenet_v2.MobileNetV2",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 0.5}
    },
    "mobilenet_v2_0_35": {
        "load_name": "mobilenet_v2.MobileNetV2",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 0.35}
    },
    "mobilenet_v1_1_00": {
        "load_name": "mobilenet.MobileNet",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 1}
    },
    "mobilenet_v1_0_75": {
        "load_name": "mobilenet.MobileNet",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_7_5_224_tf_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 0.75}
    },
    "mobilenet_v1_0_50": {
        "load_name": "mobilenet.MobileNet",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_5_0_224_tf_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 0.50}
    },
    "mobilenet_v1_0_25": {
        "load_name": "mobilenet.MobileNet",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_224_tf_no_top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": {"alpha": 0.25}
    },
    "resnet152_v2": {
        "load_name": "resnet_v2.ResNet152V2",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "resnet101_v2": {
        "load_name": "resnet_v2.ResNet101V2",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "resnet50_v2": {
        "load_name": "resnet_v2.ResNet50V2",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "resnet152_v1": {
        "load_name": "resnet.ResNet152",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "resnet101_v1": {
        "load_name": "resnet.ResNet101",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "resnet50_v1": {
        "load_name": "resnet.ResNet50",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "vgg19_bn": {
        "load_name": "vgg19.VGG19",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "vgg16_bn": {
        "load_name": "vgg16.VGG16",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "densenet201": {
        "load_name": "densenet.DenseNet201",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "densenet169": {
        "load_name": "densenet.DenseNet169",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    "densenet121": {
        "load_name": "densenet.DenseNet121",
        "weights": "https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    },
    # "efficientnet_b0": {
    #     "load_name": "efficientnet.EfficientNetB0",
        
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    # "efficientnet_b1": {
    #     "load_name": "efficientnet.EfficientNetB1",
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb1_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    # "efficientnet_b2": {
    #     "load_name": "efficientnet.EfficientNetB2",
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb2_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    # "efficientnet_b3": {
    #     "load_name": "efficientnet.EfficientNetB3",
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    # "efficientnet_b4": {
    #     "load_name": "efficientnet.EfficientNetB4",
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb4_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    # "efficientnet_b5": {
    #     "load_name": "efficientnet.EfficientNetB5",
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb5_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    # "efficientnet_b6": {
    #     "load_name": "efficientnet.EfficientNetB6",
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb6_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    # "efficientnet_b7": {
    #     "load_name": "efficientnet.EfficientNetB7",
    #     "weights": "https://storage.googleapis.com/keras-applications/efficientnetb7_notop.h5",
    #     "default_input_shape": (224, 224, 3),
    # "default_output_shape": (1000, ),    
    # "kwargs": None
    # },
    "nasnet_large": {
        "load_name": "nasnet.NASNetLarge",
        "weights": "https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5",
        "default_input_shape": (331, 331, 3),
        "default_output_shape": (1000, ),
        "kwargs": None        
    },
    "nasnet_mobile": {
        "load_name": "nasnet.NASNetMobile",
        "weights": "https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-mobile-no-top.h5",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
        "kwargs": None
    }    
}


import sys
import numpy as np
import os

from conductor._base import get_conductor_path
from conductor.utils import download
# from tvm.contrib.download import download_testdata
from tvm import relay

DEFAULT_INPUT_NAMES = ["input_1"]

def import_keras():
    stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        import tensorflow as tf
        from tensorflow import keras
        return tf, keras
    finally:
        sys.stderr = stderr

def get_keras_input(model):
    tf, _ = import_keras()
    in_shapes = []
    for layer in model._input_layers:
        if tf.executing_eagerly():
            in_shapes.append(tuple(dim if dim is not None else 1 for dim in layer.input.shape))
        else:
            in_shapes.append(tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape))
    inputs = [np.random.uniform(size=shape, low=-1.0, high=1.0) for shape in in_shapes]
    shape_dict = {name: x.shape for (name, x) in zip(model.input_names, inputs)}
    return inputs, shape_dict

def is_sequential_p(model):
    _, keras = import_keras()
    return isinstance(model, keras.models.Sequential)

def sequential_to_functional(model):
    _, keras = import_keras()
    assert is_sequential_p(model)
    input_layer = keras.layers.Input(
        batch_shape=model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in model.layers:
        prev_layer = layer(prev_layer)
    model = keras.models.Model([input_layer], [prev_layer])
    return model

def acquire(model_name, details, custom_path=None, force_reload=False):
    import tensorflow.python.keras.applications as keras_apps
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
    model_pth = os.path.join(models_pth, model_name + "_" + "keras" + ".cond.framework")
    os.makedirs(model_pth, exist_ok=True)
    model = keras_apps
    parts = details["load_name"].split(".")
    for part in parts:
        model = getattr(model, part)
    model_params_path = os.path.join(model_pth, details["weights"].split("/")[-1])
    if not os.path.exists(model_params_path) or force_reload:
        return (model, download(details["weights"], model_params_path))
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

    if in_shape is not None:
        input_shape = in_shape
    else:
        input_shape = details["default_input_shape"]

    kwargs = details.get("kwargs", None)
    _kwargs = {"include_top": False, "weights": None, "input_shape": input_shape}
    if kwargs:
        merged_kwargs = {**kwargs, **_kwargs}
    else:
        merged_kwargs = _kwargs

    model, model_params_path = acq_result
    model = model(**merged_kwargs)
    model.load_weights(model_params_path)

    if is_sequential_p(model):
        model = sequential_to_functional(model)
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
    
    inputs, shape_dict = get_keras_input(model_instance)
    return (relay.frontend.from_keras(model_instance, shape_dict, layout="NHWC"), {
        "input_name": [_in for _in in model_instance.input_names],
        "batch_size": batch_size,
        "input_shape": [_sh.shape for _sh in inputs],
        "output_shape": output_shape
    })


