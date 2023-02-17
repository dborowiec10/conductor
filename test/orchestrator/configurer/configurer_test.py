from conductor.mediation import encode_measure_input_result, decode_measure_input_result
from conductor.tensor_program import TensorProgram

import os

dir_path = os.path.dirname(os.path.realpath(__file__))


tp = TensorProgram("batch_matmul_51268830.cuda", None, None, None, None, None, None)

print(tp)

with open(os.path.join(dir_path, "ansor_logs.log"), "r") as _fil:
    for r in _fil:
        print(decode_measure_input_result(r))


