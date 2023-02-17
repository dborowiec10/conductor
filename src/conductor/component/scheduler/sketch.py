import traceback
from conductor.mediation import MeasureInput, Tasker, SingleConfigContext, MeasureErrorNo, ERROR_TYPES
from conductor.component.scheduler._base import Scheduler
from conductor.utils import make_traceback_info
from conductor.configurer.configurer import Configurer
from conductor.mediation import MeasureInput as CondMeasureInput, MeasureResult as CondMeasureResult

from tvm.auto_scheduler.measure_record import state_to_string
from tvm.auto_scheduler.measure import MeasureResult
import tvm
import cloudpickle
import numpy as np
import logging

logger = logging.getLogger("conductor.component.scheduler.sketch")

class SketchScheduler(Scheduler):
    _name = "sketch"
    
    def __repr__(self):
        return Scheduler.__repr__(self) + ":" + SketchScheduler._name

    def __init__(self):
        Scheduler.__init__(self)

    def get_measure_result(self, costs, error_no, error_msg, time_cost, timestamp, achieved_flop, mean, total_time, status):
        return MeasureResult(costs, error_no, error_msg, time_cost, timestamp, achieved_flop, mean=mean, total_time=total_time)

    def measure_input_to_config(self, inp):
        return inp

    def get_mean_cost(self, res):
        if isinstance(res, MeasureResult):
            if isinstance(res.costs[0], tvm.tir.FloatImm):
                return np.mean([float(c.value) for c in res.costs])
            else:
                return np.mean([float(c) for c in res.costs])
        else:
            logger.error("unable to retrieve mean for res: %s", str(res))
            return 1e20

    def get_build_inputs(self, task, configs, device_ids, dev_ctx_details, hash_callback, options=None):
        theoretical_flop = Tasker.task_theoretical_flop(task)
        unique_dev_ids = list(set(device_ids))
        sm_map = {k: self.get_sm(dev_ctx_details, task.target, k) for k in unique_dev_ids}
        build_inputs = []
        temp_configs = []
        for k, (i, dev_id) in enumerate(zip(configs, device_ids)):
            temp_config = {"config": i.state, "config_repr": state_to_string(i.state), "flop": theoretical_flop}
            temp_configs.append(temp_config)
            build_inputs.append({
                "orch_scheduler": self,
                "task": "magic",
                "config": {"config": None, "config_repr": state_to_string(i.state), "flop": theoretical_flop},
                "measure_input": cloudpickle.dumps(i.serialize()),
                "options": {
                    "options": options,
                    "idx": k
                },
                "sm": sm_map[dev_id],
                "verify": self.get_build_options(dev_ctx_details, task.target, dev_id),
                "hash_callback": hash_callback
            })

        return build_inputs, temp_configs, theoretical_flop


    def get_inp_res_err(self, configs, results, theoretical_flop, task):
        tot_error_count = 0
        error_counts = [0 for i in range(len(ERROR_TYPES))]
        m_inps = []
        m_res = []
        c_inps = []
        c_res = []
        for k, (conf, res) in enumerate(zip(configs, results)):
            if res.error_no != MeasureErrorNo.NO_ERROR:
                tot_error_count += 1
            error_counts[res.error_no] += 1
            m_inps.append(conf)
            m_res.append(res)
            c_inps.append(CondMeasureInput(task, task.workload_key, task.target, conf, theoretical_flop))
            c_res.append(CondMeasureResult(
                res.costs, 
                res.error_no, 
                res.error_msg, 
                res.all_cost, 
                res.timestamp, 
                res.achieved_flop,
                res.mean,
                res.total_time,
                None,
                other=res
            ))

        return m_inps, m_res, c_inps, c_res, tot_error_count, error_counts


    # function will provide a working schedule in most cases
    # sketch functionality requires some tuning to occur before a working schedule can be established
    # this is because the threads need to be matched to the axes for cuda environment
    # in case it fails, try increasing trials or timeout
    def from_tensor_program_baseline(self, tensor_program, options=None):
        from tvm.ir.transform import PassContext
        from tvm.driver.build_module import build
        from tvm import device
        from tvm.runtime.ndarray import array as tvm_array_constr
        from conductor.utils import get_const_tuple, generate_tensor_data
        from conductor.compiler._target import create_target
        _config = options.pass_configurations

        def build_tp(schedule, args):
            target, _ = create_target(options.target, options.target_host, options.target_opts)
            with target:
                with PassContext(config=_config):
                    return build(schedule, args, target=target)

        def eval_tp(module, args, eval_options, evaluator):
            dev = device(eval_options.device_type, eval_options.device_id)
            tens_data = []
            for x in args:
                arr = tvm_array_constr(generate_tensor_data(get_const_tuple(x.shape), x.dtype, eval_options.fill_mode), dev)
                tens_data.append(arr)
            costs, mean, total_time, other = evaluator.evaluate(
                module,
                module.entry_name,
                None,
                dev,
                tens_data
            )
            return ((costs, mean, total_time, other), None)

        tsk = Tasker.task_from_tensor_program(tensor_program, "sketch")
        from conductor.component.search_policy.sketch import SketchPolicy
        from conductor.component.measurer.stub import StubMeasurer
        from conductor.executor.tensor_program_executor_routine import execute_routine
        from conductor.executor.options import ExecutionOptions

        from conductor.component.evaluator.default import DefaultEvaluator
        from tvm.auto_scheduler.cost_model import RandomModel

        sh, args = None, None
        error_no = None
        error_msg = None

        # This is really important
        # we need to fake a measurer to capture schedule configs proposed by the sketch policy
        # we are using a random model and only initializing the sketch policy with N random samples
        # we don't care too much about the result, as long as it is a valid schedule
        # this is the next best thing for a "default" schedule in ansor as it is incapable of producing those
        # without all this faff... bad design guys
        meas = StubMeasurer()

        search_policy = SketchPolicy(tsk, RandomModel(), configs={
            "eps_greedy": 0.01,
            "sample_init_min_population": 100,
            "evolutionary_search_num_iters": 0,
        })
        try:
            _ = search_policy.continue_search_one_round(100, meas.program_measurer)
        except Exception:
            # this will generate an error as the stub measurer returns empty arrays (which c++ policy cannot handle)
            # ignore this, measurer should have hijacked the schedule configs by now
            pass

        for p in meas.proposed:
            try:
                _sh, _args, error_no, error_msg = self.from_task(tsk, p.state, tsk.target)
                opts_exec = ExecutionOptions()
                eval = DefaultEvaluator({
                    "num_avg_runs": 3,
                    "num_avg_repeat": 20
                })
                outfunc = build_tp(_sh, _args)
                if outfunc is not None:
                    out = eval_tp(outfunc, _args, opts_exec, eval)
                    if out[1] is None:
                        print(out[0])
                        sh = _sh
                        args = _args
                        break

            except Exception as e:
                pass

        return sh, args, error_no, error_msg

    def from_tensor_program_context(self, tensor_program, context):
        try:
            tsk = Tasker.task_from_tensor_program(tensor_program, "sketch")
            config = context.query(tsk.target, tsk.workload_key)
            if config is not None:
                schedule, args, error_no, error_msg = self.from_task(tsk, config, tsk.target, options=None)
            else:
                schedule = None
                args = None
                error_no = MeasureErrorNo.INSTANTIATION_ERROR
                error_msg = "Config is None/invalid!"
        except Exception:
            schedule = None
            args = None
            error_no = MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = traceback.format_exc()
        return schedule, args, error_no, error_msg

    def from_tensor_program_logfile(self, tensor_program, logfile):
        try:
            tsk = Tasker.task_from_tensor_program(tensor_program, "sketch")
            configurer = Configurer(logfile)
            configurer.load()
            config = configurer.pick_best_config(tsk.target, tsk.workload_key, None)
            if config is not None:
                schedule, args, error_no, error_msg = self.from_task(tsk, config, tsk.target, options=None)
            else:
                schedule = None
                args = None
                error_no = MeasureErrorNo.INSTANTIATION_ERROR
                error_msg = "Config is None/invalid!"
        except Exception:
            schedule = None
            args = None
            error_no = MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = make_traceback_info()
        return schedule, args, error_no, error_msg

    # generate schedule given a tensor_program object and config
    def from_tensor_program_config(self, tensor_program, config, options=None):
        tsk = Tasker.task_from_tensor_program(tensor_program, "sketch")
        return self.from_task(tsk, config, tensor_program.target)

    # generate schedule given task and config
    # typically used by tuning methods to instantiate candidates
    def from_task(self, task, config, target, options=None):
        try:
            if isinstance(target, str):
                target = tvm.target.Target(target)
            with target:
                with SingleConfigContext(config):
                    schedule, args = task.compute_dag.apply_steps_from_state(config, layout_rewrite=task.layout_rewrite_option)
            error_no = MeasureErrorNo.NO_ERROR
            error_msg = ""
        except Exception:
            error_no = MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = make_traceback_info()
        return schedule, args, error_no, error_msg

    def from_autoschedule_topi(self, func_name, coded_key, decoded_key, target, io_tensors, args, dag, input_map, config, has_layout_free, has_complex_op):
        if config is None:
            return None
        schedule, _ = dag.apply_steps_from_state(config)
        return schedule