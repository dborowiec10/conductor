from conductor.mediation import SingleConfigContext, Tasker, MeasureErrorNo, flex_wrapper
from conductor.configurer.configurer import Configurer
from conductor.component.scheduler._base import Scheduler
from conductor.utils import make_traceback_info
from conductor.component.method.flex.scheduler import schedule_with_config
from conductor.component.method.flex.task import Task as FlexTask
import tvm
import traceback
import logging
import cloudpickle
logger = logging.getLogger("conductor.component.scheduler.flex")

class FlexScheduler(Scheduler):
    _name = "flex"
    
    def __repr__(self):
        return Scheduler.__repr__(self) + ":" + FlexScheduler._name

    def __init__(self):
        Scheduler.__init__(self)

    def get_build_inputs(self, task, configs, device_ids, dev_ctx_details, hash_callback, options=None):
        theoretical_flop = Tasker.task_theoretical_flop(task)
        unique_dev_ids = list(set(device_ids))
        sm_map = {k: self.get_sm(dev_ctx_details, task.target, k) for k in unique_dev_ids}
        build_inputs = []
        temp_configs = []
        for k, (c, dev_id) in enumerate(zip(configs, device_ids)):
            temp_config = {"config": c, "config_repr": str(c), "flop": theoretical_flop}
            temp_configs.append(temp_config)
            build_inputs.append({
                "orch_scheduler": self,
                "task": "flex",
                "measure_input": cloudpickle.dumps(task),
                "config": temp_config,
                "options": {
                    "options": options,
                    "idx": k
                },
                "sm": sm_map[dev_id],
                "verify": self.get_build_options(dev_ctx_details, task.target, dev_id),
                "hash_callback": hash_callback
            })
        return build_inputs, temp_configs, theoretical_flop

    # generate a baseline schedule that is correct and will execute on the device
    # performance of the schedule is unpredictable
    def from_tensor_program_baseline(self, tensor_program):
        pass

    def from_tensor_program_logfile(self, tensor_program, logfile, options=None):
        configurer = Configurer(logfile)
        configurer.load()
        task = Tasker.task_from_tensor_program(tensor_program, "flex")
        config = configurer.pick_best_config(task.target, task.key, task.args, t_kwargs={})
        return self.from_task(task, config, task.target, options=options)

    # generate schedule given a tensor_program object and config
    def from_tensor_program_config(self, tensor_program, config, options=None):
        if options == None:
            return None, None, MeasureErrorNo.INSTANTIATION_ERROR, "options not provided!"
        task = Tasker.task_from_tensor_program(tensor_program, "flex")
        return self.from_task(task, config, tensor_program.target, options=options)

    # generate schedule given task and config
    # typically used by tuning methods to instantiate candidates
    def from_task(self, task, config, target, options=None):
        if options["options"] == None:
            return None, None, MeasureErrorNo.INSTANTIATION_ERROR, "options not provided!"
        op_pos, rewrite = options["options"]
        try:
            if isinstance(target, str):
                target = tvm.target.Target(target)

            with SingleConfigContext(config):
                with target:
                    schedule, args  = schedule_with_config(
                        task,
                        config,
                        op_pos=op_pos,
                        rewrite=rewrite
                    )
            error_no = MeasureErrorNo.NO_ERROR
            error_msg = ""
        except Exception as e:
            schedule = None
            args = None
            error_no = MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = traceback.format_exc()

        return schedule, args, error_no, error_msg

    def from_autoschedule_topi(self, coded_key, target, io_tensors, args, dag, config):
        if config is None:
            return None
        task = FlexTask(coded_key, coded_key, flex_wrapper(io_tensors), args, target, None)
        schedule, _ = schedule_with_config(
            task,
            config,
            op_pos=None,
            rewrite=False
        )
        return schedule