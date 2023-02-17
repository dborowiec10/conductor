from conductor import mediation
import os
import tvm
from conductor.model import Model
from conductor.workloads.models.tvmmods import models, acquire, convert, instantiate
from pprint import pprint

model = Model("mlp", "tvm")

print(model)

target = "cuda"
target_host = "llvm"

module = model.model_json

if isinstance(target, str):
    target = tvm.target.Target(target)
if isinstance(target_host, str):
    target_host = tvm.target.Target(target_host)
if isinstance(module, tvm.relay.function.Function):
    module = tvm.IRModule.from_expr(module)

print(target)
print(target_host)
print(module)


print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")
print("------------------------------------------------------")

trace_env = mediation.TensorProgramExtractionEnvironment()

with tvm.autotvm.tophub.context(target):
    with trace_env:
        mediation.trace_module(module, model.model_params, target)


templateable = []
untemplateable = []

for k, d in trace_env.tensor_program_definitions.items():
    if d["schedule_gen_func"] is not None:
        templateable.append((k, d))
    else:
        untemplateable.append((k, d))

print("TEMPLATEABLE")
pprint("========================================")
for k, d in templateable:
    pprint(k)
    pprint(d)
    pprint("----------------------------------------")


pprint("========================================")

print("UNTEMPLATEABLE")
pprint("========================================")
for k, d in untemplateable:
    pprint(k)
    pprint(d)
    pprint("----------------------------------------")

# pprint(trace_env.tensor_program_definitions)
