
import tvm
from conductor.component.method.flex.space import EnumSpace
from conductor.component.method.flex.nn import conv2d_nchwc

TASK_TABLE = {}

class Task(object):
    def __init__(self, category, name, func, args, target, target_host, dev_id=0):
        self.key = "{}:{}:{}".format(category, name, args)#
        self.func = func
        self.args = args
        self.target = target
        self.target_host = target_host
        self.dev_id = dev_id
        self.category = category
        self.special_space = {}

    def set_specific_space(self, key, type, knobs):
        if type == "enum":
            self.special_space[key] = EnumSpace(knobs)
        else:
            raise RuntimeError("Not supported space type: %s" % type)

def register_compute(key, compute, override=False):
    if key in TASK_TABLE and not override:
        pass
    TASK_TABLE[key] = compute

def register_task(task, override=False):
    TASK_TABLE[task.key] = task

def conv2d_nchwc_layout(N, C, H, W, K, k=3, st=1, pad=0, dilation=1, group=1, vlen1=8, vlen2=8):
    use_bias = False
    inputs = tvm.te.placeholder([N, C // vlen1 // group, H, W, vlen1], dtype="float32")
    weight = tvm.te.placeholder([K // vlen2, C // vlen1 // group, k, k, vlen1, vlen2], dtype="float32")
    if use_bias:
        bias = tvm.te.placeholder([K // vlen2, vlen2], dtype="float32")
    else:
        bias = None 
    output = conv2d_nchwc(inputs, weight, bias, stride=st, padding=pad, dilation=dilation, groups=group)
    if use_bias:
        return [output.op], [inputs, weight, bias, output]
    else:
        return [output.op], [inputs, weight, output]