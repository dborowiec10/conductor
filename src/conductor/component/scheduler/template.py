from conductor.mediation import Tasker, MeasureErrorNo, SingleConfigContext, FallbackContext
from conductor.component.scheduler._base import Scheduler
from conductor.utils import make_traceback_info
from conductor.configurer.configurer import Configurer

import tvm
from tvm.autotvm.task.task import compute_flop

import logging

logger = logging.getLogger("conductor.component.scheduler.template")

class TemplateScheduler(Scheduler):
    _name = "template"
    
    def __repr__(self):
        return Scheduler.__repr__(self) + ":" + TemplateScheduler._name

    def __init__(self):
        Scheduler.__init__(self)

    def from_tensor_program_baseline(self, tensor_program):
        if tensor_program.is_templateable():
            try:
                task = Tasker.task_from_tensor_program(tensor_program, "template")
                if isinstance(tensor_program.target, str):
                    target = tvm.target.Target(tensor_program.target)
                with FallbackContext():
                    with target:
                        schedule, args = task.func(*task.args)                    
                error_no = MeasureErrorNo.NO_ERROR
                error_msg = ""
            except Exception:
                schedule = None
                args = None
                error_no = MeasureErrorNo.INSTANTIATION_ERROR
                error_msg = make_traceback_info()
            return schedule, args, error_no, error_msg
        else:
            return None, None, MeasureErrorNo.INSTANTIATION_ERROR, "tensor_program is not templateable!"

    def from_tensor_program_context(self, tensor_program, context):
        if tensor_program.is_templateable():
            try:
                task = Tasker.task_from_tensor_program(tensor_program, "template")
                if isinstance(tensor_program.target, str):
                    target = tvm.target.Target(tensor_program.target)
                else:
                    target = tensor_program.target
                with context:
                    with target:
                        schedule, args = task.func(*task.args, **task.kwargs)
                error_no = 0
                error_msg = ""
            except Exception:
                schedule = None
                args = None
                error_no = MeasureErrorNo.INSTANTIATION_ERROR
                error_msg = make_traceback_info()
            return schedule, args, error_no, error_msg
        else:
            return None, None, MeasureErrorNo.INSTANTIATION_ERROR, "tensor_program is not templateable!"        

    # generate schedule given a tensor_program object and logfile
    def from_tensor_program_logfile(self, tensor_program, logfile):
        if tensor_program.is_templateable():
            try:
                configurer = Configurer(logfile)
                task = Tasker.task_from_tensor_program(tensor_program, "template")
                configurer.load()
                config = configurer.pick_best_config(task.target, task.workload, task.args)
                schedule, args, error_no, error_msg = self.from_task(task, config, task.target, options=None)
            except Exception:
                schedule = None
                args = None
                error_no = MeasureErrorNo.INSTANTIATION_ERROR
                error_msg = make_traceback_info()
            return schedule, args, error_no, error_msg
        else:
            return None, None, MeasureErrorNo.INSTANTIATION_ERROR, "tensor_program is not templateable!"

    # generate schedule given a tensor_program object and config
    def from_tensor_program_config(self, tensor_program, config, options=None):
        task = Tasker.task_from_tensor_program(tensor_program, "template")
        return self.from_task(task, config, tensor_program.target)

    # generate schedule given task and config
    # typically used by tuning methods to instantiate candidates
    def from_task(self, task, config, target, options=None):
        # prepare target if necessary
        if isinstance(target, str):
            target = tvm.target.Target(target)
        with SingleConfigContext(config):
            with target:
                sch, args = task.func(*task.args, **task.kwargs)

        if not task.flop or not config.flop:
            config.flop = task.flop = compute_flop(sch)
    
        if not config.valid():
            error_no = MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = " :: ".join(config.errors)
        else:
            error_no = MeasureErrorNo.NO_ERROR
            error_msg = ""
        return sch, args, error_no, error_msg
