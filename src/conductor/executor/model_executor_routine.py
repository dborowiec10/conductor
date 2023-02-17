import traceback
import json
import numpy as np
from tvm import device
from tvm.contrib.graph_executor import create
from conductor.utils import get_input_info, make_inputs_dict

def execute_routine(_input):
    comp_model, options, evaluator = _input

    try:
        module = comp_model.get_lib()
        j = comp_model.get_json()
        p = comp_model.get_params()

        dev = device(options.device_type, options.device_id)
        
        m = create(j, module, dev)

        shape_dict, dtype_dict = get_input_info(json.loads(j), p)
        inputs_dict = make_inputs_dict(shape_dict, dtype_dict, None, options.fill_mode)
        m.set_input(**inputs_dict)

        costs, mean, total_time, other = evaluator.evaluate(
            m.module,
            "run", 
            None, 
            dev, 
            []
        )

        return ((costs, mean, total_time, other), None)

    except Exception as e:
        return ([None, None, None, None], traceback.format_exc())