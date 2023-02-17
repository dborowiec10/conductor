from conductor.workloads.tensor_programs.generator import generate_tensor_programs
from pprint import pprint

ret = generate_tensor_programs(save=True)
pprint(ret)