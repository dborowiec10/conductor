
models = {
    "mnasnet0_50": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_0.50_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mnasnet0_75": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_0.75_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mnasnet1_00": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mnasnet1_00_96": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_96/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mnasnet1_00_128": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_128/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mnasnet1_00_160": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_160/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mnasnet1_00_192": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_192/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mnasnet1_30": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.3_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25_128_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_128_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_224_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
# "efficientnet_b0": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b0_quantized_int8": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/int8/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b1": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite1/fp32/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b1_quantized_int8": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite1/int8/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b2": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite2/fp32/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b2_quantized_int8": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite2/int8/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# }, 
# "efficientnet_b3": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite3/fp32/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b3_quantized_int8": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite3/int8/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b4": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite4/fp32/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
# "efficientnet_b4_quantized_int8": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite4/int8/2?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
    "mobilenet_v1_0_25": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25_128": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_128/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25_160": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_160/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25_160_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_160_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25_192": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_192/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_25_192_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_192_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_224_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50_128": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_128/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50_128_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_128_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50_160": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_160/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50_160_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_160_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50_192": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_192/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_50_192_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.50_192_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_224_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75_128": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_128/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75_128_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_128_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75_160": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75_160_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75_192": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_192/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_0_75_192_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_192_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00_128": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_128/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00_128_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_128_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (128, 128, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00_160": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_160/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00_160_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_160_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (160, 160, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00_192": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_192/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00_192_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_192_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (192, 192, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v1_1_00_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v2_1_00": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v2_1_00_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224_quantized/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, ),
    },
    "mobilenet_v3_0_75_small": {
        "model": "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_075_224/classification/5/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v3_0_75_large": {
        "model": "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_large_075_224/classification/5/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v3_1_00_small": {
        "model": "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/classification/5/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "mobilenet_v3_1_00_large": {
        "model": "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_large_100_224/classification/5/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "nasnet_large": {
        "model": "https://tfhub.dev/tensorflow/lite-model/nasnet/large/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "nasnet_mobile": {
        "model": "https://tfhub.dev/tensorflow/lite-model/nasnet/mobile/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "inception_resnet_v2": {
        "model": "https://tfhub.dev/tensorflow/lite-model/inception_resnet_v2/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
# "inception_v1_quantized": {
#     "model": "https://tfhub.dev/tensorflow/lite-model/inception_v1_quant/1/metadata/1?lite-format=tflite",
#     "default_input_shape": (224, 224, 3),
#     "default_output_shape": (1000, )
# },
    "inception_v2_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/inception_v2_quant/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "inception_v3": {
        "model": "https://tfhub.dev/tensorflow/lite-model/inception_v3/1/metadata/1?lite-format=tflite",
        "default_input_shape": (299, 299, 3),
        "default_output_shape": (1000, )
    },
    "inception_v3_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/inception_v3_quant/1/metadata/1?lite-format=tflite",
        "default_input_shape": (299, 299, 3),
        "default_output_shape": (1000, )
    },
    "inception_v4": {
        "model": "https://tfhub.dev/tensorflow/lite-model/inception_v4/1/metadata/1?lite-format=tflite",
        "default_input_shape": (299, 299, 3),
        "default_output_shape": (1000, )
    },
    "inception_v4_quantized": {
        "model": "https://tfhub.dev/tensorflow/lite-model/inception_v4_quant/1/metadata/1?lite-format=tflite",
        "default_input_shape": (299, 299, 3),
        "default_output_shape": (1000, )
    },
    "resnet101_v2": {
        "model": "https://tfhub.dev/tensorflow/lite-model/resnet_v2_101/1/metadata/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (1000, )
    },
    "fungi_mobile_v1": {
        "model": "https://tfhub.dev/svampeatlas/lite-model/vision/classifier/fungi_mobile_V1/1/default/1?lite-format=tflite",
        "default_input_shape": (299, 299, 3),
        "default_output_shape": (4128, )
    },
    "popular_us_products_v1": {
        "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/popular_us_products_V1/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (100000, )
    },
    "popular_wine_v1": {
        "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/popular_wine_V1/1?lite-format=tflite",
        "default_input_shape": (224, 224, 3),
        "default_output_shape": (409776, )
    },
# "landmarks_south_america_v1": {
#     "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/landmarks_classifier_south_america_V1/1?lite-format=tflite",
#     "default_input_shape": (321, 321, 3),
#     "default_output_shape": (99206, )
# },
# "landmarks_north_america_v1": {
#     "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/landmarks_classifier_north_america_V1/1?lite-format=tflite",
#     "default_input_shape": (321, 321, 3),
#     "default_output_shape": (99424, )
# },
# "landmarks_africa_v1": {
#     "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/landmarks_classifier_africa_V1/1?lite-format=tflite",
#     "default_input_shape": (321, 321, 3),
#     "default_output_shape": (19293, )
# },
# "landmarks_oceania_antarctica_v1": {
#     "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/landmarks_classifier_oceania_antarctica_V1/1?lite-format=tflite",
#     "default_input_shape": (321, 321, 3),
#     "default_output_shape": (56707, )
# },
# "landmarks_europe_v1": {
#     "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/landmarks_classifier_europe_V1/1?lite-format=tflite",
#     "default_input_shape": (321, 321, 3),
#     "default_output_shape": (99125, )
# },
# "landmarks_asia_v1": {
#     "model": "https://tfhub.dev/google/lite-model/on_device_vision/classifier/landmarks_classifier_asia_V1/1?lite-format=tflite",
#     "default_input_shape": (321, 321, 3),
#     "default_output_shape": (98960, )
# }
}

DEFAULT_INPUT_NAMES = ["input"]

import os
from conductor._base import get_conductor_path
from conductor.utils import download

from tvm import relay

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

    models_pth = custom_path if custom_path is not None else os.path.join(get_conductor_path(), "models")
    model_pth = os.path.join(models_pth, model_name + "_" + "tflite" + ".cond.framework")
    os.makedirs(model_pth, exist_ok=True)
    if not os.path.exists(os.path.join(model_pth, model_name + ".tflite")) or force_reload:
        return download(details["model"], os.path.join(model_pth, model_name + ".tflite"))
    else:
        return os.path.join(model_pth, model_name + ".tflite")

def instantiate(model_name, details, acq_result, batch_size=1, in_shape=None, out_shape=None, dtype="float32"):
    import tflite.Model as model
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

    with open(acq_result, "rb") as tf_graph:
        content = tf_graph.read()

    try:
        tflite_model = model.Model.GetRootAsModel(content, 0)
    except AttributeError:
        tflite_model = model.GetRootAsModel(content, 0)

    try:
        version = tflite_model.Version()
    except Exception:
        raise RuntimeError("input file not tflite")

    if version != 3:
        raise RuntimeError("input file not tflite version 3")
    return tflite_model


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
    
    mod, params = relay.frontend.from_tflite(model_instance, shape_dict={DEFAULT_INPUT_NAMES[0]: (batch_size,) + input_shape})
    return ((mod, params), {
        "input_name": DEFAULT_INPUT_NAMES[0],
        "batch_size": batch_size,
        "input_shape": input_shape,
        "output_shape": output_shape
    })