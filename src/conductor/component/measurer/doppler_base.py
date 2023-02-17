from pprint import pformat
import numpy as np
import math
import time
import tvm
from conductor._base import Configurable
from conductor.mediation import MeasureErrorNo, Tasker
from conductor.component.measurer._base import Measurer
from conductor.component.evaluator.default import DefaultEvaluator
from conductor.experiment import Experiment

import logging

logger = logging.getLogger("conductor.component.measurer.doppler_base")

class DopplerBaseMeasurer(Measurer):
    _name = "doppler_base"
    
    def __repr__(self):
        return Measurer.__repr__(self) + ":" + DopplerBaseMeasurer._name

    def __init__(self, subtype, builder, runner, configs=None, child_default_configs={}):
        Measurer.__init__(self, subtype, builder, runner, configs=configs, child_default_configs=Configurable.merge_configs({
            "timeout_k": 20,
            "timeout_z": 4.5,
            "timeout_s": 0.06,
            "double_mad_zscore_constant": 0.6745,
            "double_mad_zscore_threshold": 3,
            "fixed_samples": False,
            "fixed_sample_count": 0,
            "rank_remeasure_perc": 0.1,
            "rank_remeasure_timeout": 6,
            "perc_remeasure_samples": 0.20,
            "error_to_reset": 0.75,
            "fixed_timeout": False
        }, child_default_configs, override_first=True))

        self.dop = 1

        self.default_evaluator = DefaultEvaluator(configs={
            "num_avg_runs": self.runner.evaluator.number,
            "num_measure_repeat": self.runner.evaluator.repeat,
            "min_repeat_ms": self.runner.evaluator.min_repeat_ms,
            "enable_cpu_cache_flush": False if self.runner.evaluator.f_preproc == '' else True
        })

        self.task_states = {}

    def create_task_state(self, task_idx):
        if str(task_idx) not in self.task_states:
            self.task_states[str(task_idx)] = self.task_state_template.copy()

    def measure_rank(self, task, task_idx, measure_inputs, measure_results):
        Experiment.current.set_experiment_stage("measurer:rank")
        strime = time.time()

        self.runner.n_parallel = 1
        self.runner.load_one_per_dev(self.num_devs)

        if self.using_mps == "mps" and self.mps is not None:
            logger.info("STOPPING MPS FOR RANK REMEASURE")
            logger.info("ENABLING DEFAULT PROGRAM EVALUATOR FOR RANK REMEASURE")
            self.mps.stop(self.mps_dir)
        self._prev_evaluator = self.runner.evaluator
        self.runner.evaluator = self.default_evaluator

        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = True
        combined = list(zip(measure_inputs, self.task_states[str(task_idx)]["build_results"], measure_results))
        combined.sort(key=lambda s: self.builder.orch_scheduler.get_mean_cost(s[2]))
        rank_num = int(np.ceil(len(combined) * self.config["rank_remeasure_perc"]))

        all_results = []
        configs = []
        build_results = []

        for k, (i, br, r) in enumerate(combined[:rank_num]):
            logger.info("measurer: orig performance ranking ## [%d]: %f", k, self.builder.orch_scheduler.get_mean_cost(r))
            configs.append(self.builder.orch_scheduler.measure_input_to_config(i))
            build_results.append(br)

        # TODO: THEN HERE, need to split configs across available devs
        _, updated_configs, _ = self.builder.orch_scheduler.get_build_inputs(
            task,
            configs,
            [self.device_ids[0]] * len(configs),
            self.runner.dev_ctx_details,
            self.hash_callback,
            options=None
        )

        self.timings["calc_rank_sampling"] += time.time() - strime

        strrunime = time.time()
        all_results, _ = self.run_configs(
            updated_configs, 
            build_results, 
            self.spread_across(len(build_results), self.device_ids), 
            n_parallel=1,
            remove_schedules=True, 
            run_type="rank_remeasure", 
            set_timeout=self.determine_timeout(self.config["rank_remeasure_timeout"], num_devs=self.num_devs)
        )

        self.timings["running_cand_rank"] += time.time() - strrunime

        strime = time.time()

        for k, r in enumerate(all_results):
            logger.info("measurer: remeasured performance ranking ## [%d]: %f", k, self.builder.orch_scheduler.get_mean_cost(r))

        theoretical_flop = Tasker.task_theoretical_flop(task)
        _, _, conf_m_inputs, conf_m_results, _, error_counts = self.builder.orch_scheduler.get_inp_res_err(configs, all_results, theoretical_flop, task)
        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = False
        self.configurer.add_ranked_records(conf_m_inputs, conf_m_results)
        self.runner.evaluator = self._prev_evaluator
        logger.info("measurer stop on #%s inputs, total errors: %s", str(len(configs)), str(sum(error_counts[1:])))

        self.timings["calc_rank_sampling"] += time.time() - strime

        logger.info(pformat(self.timings))

    def spread_across(self, num_cands, device_ids):
        dev_indexes = []
        dev_cnt = 0
        for i in range(num_cands):
            dev_indexes.append(device_ids[dev_cnt])
            dev_cnt += 1
            if dev_cnt == len(device_ids):
                dev_cnt = 0
        return dev_indexes

    # https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    def doubleMADsfromMedian(self, y, indexes):
        with np.errstate(divide='ignore'):
            m = np.median(y)
            abs_dev = np.abs(y - m)
            left_mad = np.median(abs_dev[y <= m])
            right_mad = np.median(abs_dev[y >= m])
            y_mad = left_mad * np.ones(len(y))
            y_mad[y > m] = right_mad
            modified_z_score = self.config["double_mad_zscore_constant"] * abs_dev / y_mad
            modified_z_score[y == m] = 0
        return (np.array(indexes)[modified_z_score > self.config["double_mad_zscore_threshold"]].tolist(), m)

    def determine_timeout(self, cur_dop, num_devs=1):
        if not self.config["fixed_timeout"]:
            cd = math.ceil(num_devs * 0.5) * cur_dop
            k = self.config["timeout_k"] # max timeout parameter (user specified)
            z = self.config["timeout_z"] # lower bound of timeout
            s = self.config["timeout_s"] # steepness parameter (how quickly to ramp up timeout once lower bound reached)
            alpha = (float(k/2) - float(z/5))
            beta = (float(k/2) + float(z/5))
            return int(math.floor((alpha * math.tanh((cd - (z ** 2)) * s)) + beta)) + 1
        else:
            return self.runner.timeout

    def pick_n_configs(self, all_configs, n):
        # pick self.dop number of configs
        curr_inp_configs = all_configs[0:n]
        curr_configs = [uc for (inp, uc) in curr_inp_configs]
        curr_inputs = [inp for (inp, uc) in curr_inp_configs]
        return (all_configs[n:], curr_configs, curr_inputs)

    def build_configs(self, inputs):
        self.builder.set_n_parallel(len(inputs))
        outs = self.builder.build(inputs)
        Experiment.current.set_experiment_stage("measurer:measure")
        return outs

    def run_configs(self, configs, bres, device_indices, n_parallel=None, remove_schedules=True, run_type="regular", set_timeout=None):
        self.runner.n_parallel = n_parallel
        self.runner.remove_built_schedule = remove_schedules
        self.runner.run_type = run_type
        if set_timeout is not None:
            logger.info("measurer: setting timeout to %d" % set_timeout)
            self.runner.timeout = set_timeout
        part_res = self.runner.run(configs, bres, device_indices)
        Experiment.current.set_experiment_stage("measurer:measure" if run_type == "regular" else "measurer:rank")
        return part_res, bres

    def pick_random_success_indexes(self, measure_results, num_remeasure):
        indexes = list(range(len(measure_results)))
        success_indexes = [idx for idx in indexes if measure_results[idx].error_no == MeasureErrorNo.NO_ERROR]
        if len(success_indexes) > 1:
            chosen_indexes = np.random.choice(
                success_indexes, 
                size=num_remeasure if len(success_indexes) >= num_remeasure else len(success_indexes),
                replace=False
            )
        else:
            chosen_indexes = np.array([])
        return chosen_indexes.tolist()

    def get_window_average(self, arr, window):
        l = len(arr)
        win_index = -window if window <= l else -l
        return np.mean(np.array(arr[win_index:])) if l > 0 else 0

    def measure(self, task, configs, options=None):
        raise NotImplementedError("abstract method, needs implementation in child class!")