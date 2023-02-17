import traceback
import numpy as np
from tvm import device
from tvm.runtime.ndarray import array as tvm_array_constr
from conductor.utils import get_const_tuple, generate_tensor_data

def execute_routine(_input):
    tp, options, evaluator = _input

    try:
        module = tp.get_lib()
        dev = device(options.device_type, options.device_id)

        tens_data = []
        for x in tp.args:
            arr = tvm_array_constr(generate_tensor_data(get_const_tuple(x.shape), x.dtype, options.fill_mode), dev)
            tens_data.append(arr)

        costs, mean, total_time, other = evaluator.evaluate(
            module,
            module.entry_name,
            None,
            dev,
            tens_data
        )

        return ((costs, mean, total_time, other), None)

    except Exception as e:
        return ((None, None, None, None), traceback.format_exc())