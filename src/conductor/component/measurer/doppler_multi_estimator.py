from conductor._base import Configurable
from conductor.component.measurer.doppler_base import DopplerBaseMeasurer
from conductor.experiment import Experiment
import logging
import tvm
import time
import numpy as np
from skopt import Optimizer

logger = logging.getLogger("conductor.component.measurer.doppler_multi_estimator")

class DopplerMultiEstimatorMeasurer(DopplerBaseMeasurer):
    _name = "doppler_multi_estimator"
    
    def __repr__(self):
        return DopplerBaseMeasurer.__repr__(self) + ":" + DopplerMultiEstimatorMeasurer._name

    def __init__(self, builder, runner, configs={}, child_default_configs={}):
        DopplerBaseMeasurer.__init__(self, "doppler_multi_estimator", builder, runner, configs=configs, child_default_configs=Configurable.merge_configs({
            "max_dop": 64,
            "window_deltas": 2,
            "window_times": 2,
            "delta_threshold": 0.05,
            "failed_threshold": 0.05,
            "failed_due_to_dop_threshold": 0.25,
            "opt_update_avg_time_split": [0.8, 1.2],
            "opt_update_avg_error_split": [0.8, 1.2],
            "opt_update_avg_failed_split": [0.8, 1.2],
            "opt_time_config": {
                "base_estimator": "GBRT",
                "acq_func": "gp_hedge", 
                "acq_optimizer": "auto",
                "initial_point_generator": "random", 
                "n_initial_points": 1
            },
            "opt_error_config": {
                "base_estimator": "GBRT",
                "acq_func": "gp_hedge", 
                "acq_optimizer": "auto",
                "initial_point_generator": "random",
                "n_initial_points": 1
            },
            "opt_failed_config": {
                "base_estimator": "GBRT",
                "acq_func": "gp_hedge",
                "acq_optimizer": "auto",
                "initial_point_generator": "random",
                "n_initial_points": 1
            }
        }, child_default_configs, override_first=True))

        self.max_dop = self.config["max_dop"]
        self.window_deltas = self.config["window_deltas"]
        self.window_perc_bad = self.config["window_perc_bad"]
        self.window_times = self.config["window_times"]

        self.task_state_template = {
            "deltas_so_far": [],
            "times_so_far": [],
            "measurement_time_opt": None,
            "candidate_error_opt": None,
            "num_failed_opt": None,
            "last_it_failed": 0,
            "prior_max_dop": 0,
            "last_proposed_dop": 0,
            "dop": 1
        }

    def optimizers_propose(self):
        # querying optimizers
        st = time.time()

        mt_prop = self.task_states[str(self.task_idx)]["measurement_time_opt"].ask()[0]
        cne_prop = self.task_states[str(self.task_idx)]["candidate_error_opt"].ask()[0]
        nf_prop = self.task_states[str(self.task_idx)]["num_failed_opt"].ask()[0]

        means = [mt_prop, cne_prop, nf_prop]

        avg = np.floor(np.mean(np.array(means)))
        et = time.time()
        logger.info("optimizers_proposed_dops: TIME[%d], DELTA[%d], FAIL[%d] ... avg(%d), in %f seconds", mt_prop, cne_prop, nf_prop, avg,  et - st)
        return avg


    def optimizers_update(self, current_cand_error, current_failed, current_measurement_time):
        st = time.time()

        avg_delta = self.get_window_average(self.task_states[str(self.task_idx)]["deltas_so_far"], self.window_deltas)
        avg_perc_bad = self.get_window_average(self.task_states[str(self.task_idx)]["perc_bad_so_far"], self.window_perc_bad)
        avg_time = self.get_window_average(self.task_states[str(self.task_idx)]["times_so_far"], self.window_times)

        time_split = self.config["opt_update_avg_time_split"]
        error_split = self.config["opt_update_avg_error_split"]
        perc_bad_split = self.config["opt_update_avg_failed_split"]

        score_delta = ((avg_delta * error_split[0]) + (current_cand_error * error_split[1])) / 2
        score_perc_bad = ((avg_perc_bad * perc_bad_split[0]) + (current_failed * perc_bad_split[1])) / 2
        score_time = ((avg_time * time_split[0]) + (current_measurement_time * time_split[1])) / 2

        self.task_states[str(self.task_idx)]["last_it_failed"] = float(current_failed / self.task_states[str(self.task_idx)]["dop"])

        self.task_states[str(self.task_idx)]["measurement_time_opt"].tell([self.task_states[str(self.task_idx)]["dop"]], score_time)
        logger.info("TASK[%s] optimizer [ TIME  ] updated current(%f), avg(%f), score(%f)", str(self.task_idx), current_measurement_time, avg_time, score_time)

        self.task_states[str(self.task_idx)]["candidate_error_opt"].tell([self.task_states[str(self.task_idx)]["dop"]], score_delta)
        logger.info("TASK[%s] optimizer [ DELTA ] updated current(%f), avg(%f), score(%f)", str(self.task_idx), current_cand_error, avg_delta, score_delta)

        self.task_states[str(self.task_idx)]["num_failed_opt"].tell([self.task_states[str(self.task_idx)]["dop"]], score_perc_bad)
        logger.info("TASK[%s] optimizer [ FAIL ] updated current(%f), avg(%f), score(%f)", str(self.task_idx), current_failed, avg_perc_bad, score_perc_bad)

        et = time.time()

        logger.info("optimizers update in %f seconds", et - st)

    def measure(self, task, configs, options=None):
        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = True
        logger.info("measurer start on #%s inputs", str(len(configs)))

        # at this point, we have all build inputs prepared.
        build_inputs, updated_configs, theoretical_flop = self.builder.orch_scheduler.get_build_inputs(
            configs,
            task,
            self.runner.dev_ctx_details,
            self.runner.device_id,
            self.hash_callback,
            options=options
        )
        
        all_inp_configs = list(zip(self.build_configs(build_inputs), updated_configs))

        all_results = []

        self.create_task_state(self.task_idx)

        if self.stage == "proper":
            curr_max_dop = min(self.max_dop, len(configs))
            if curr_max_dop > self.task_states[str(self.task_idx)]["prior_max_dop"]:
                self.task_states[str(self.task_idx)]["measurement_time_opt"] = Optimizer(
                    [list(range(1, curr_max_dop + 1))], 
                    self.config["opt_time_config"]["base_estimator"], 
                    acq_func=self.config["opt_time_config"]["acq_func"], 
                    acq_optimizer=self.config["opt_time_config"]["acq_optimizer"],
                    initial_point_generator=self.config["opt_time_config"]["initial_point_generator"], 
                    n_initial_points=self.config["opt_time_config"]["n_initial_points"]
                )
                self.task_states[str(self.task_idx)]["candidate_error_opt"] = Optimizer(
                    [list(range(1, curr_max_dop + 1))], 
                    self.config["opt_error_config"]["base_estimator"], 
                    acq_func=self.config["opt_error_config"]["acq_func"], 
                    acq_optimizer=self.config["opt_error_config"]["acq_optimizer"],
                    initial_point_generator=self.config["opt_error_config"]["initial_point_generator"], 
                    n_initial_points=self.config["opt_error_config"]["n_initial_points"]
                )
                self.task_states[str(self.task_idx)]["num_failed_opt"] = Optimizer(
                    [list(range(1, curr_max_dop + 1))], 
                    self.config["opt_failed_config"]["base_estimator"], 
                    acq_func=self.config["opt_failed_config"]["acq_func"], 
                    acq_optimizer=self.config["opt_failed_config"]["acq_optimizer"],
                    initial_point_generator=self.config["opt_failed_config"]["initial_point_generator"], 
                    n_initial_points=self.config["opt_failed_config"]["n_initial_points"]
                )
                
            self.task_states[str(self.task_idx)]["prior_max_dop"] = curr_max_dop

        while len(all_inp_configs) > 0:
            if self.stage == "proper":
                self.task_states[str(self.task_idx)]["dop"] = self.optimizers_propose()
            else:
                self.task_states[str(self.task_idx)]["dop"] = 1

            prop_dop = self.task_states[str(self.task_idx)]["dop"]
            self.task_states[str(self.task_idx)]["dop"] = int(min(max(1, self.task_states[str(self.task_idx)]["dop"]), len(all_inp_configs)))
            self.task_states[str(self.task_idx)]["last_proposed_dop"] = self.task_states[str(self.task_idx)]["dop"]

            logger.info("measurer proposed_dop[%d], actual self.dop[%d], configs_left[%d]", prop_dop, self.task_states[str(self.task_idx)]["dop"], len(all_inp_configs))

            # pick n configs from the batch
            all_inp_configs, curr_configs, curr_bld_res = self.pick_n_configs(all_inp_configs, self.task_states[str(self.task_idx)]["dop"])
            
            st = time.time()
            # build & run them
            part_results, part_b_results = self.run_configs(curr_configs, curr_bld_res, remove_schedules=False, run_type="regular", set_timeout=self.determine_timeout(self.task_states[str(self.task_idx)]["dop"]))
            et = time.time()

            # initially, error is 0
            avg_delta = 0

            # percentage of bad (not errored candidates is initially 0.0)
            perc_bad = 0
            
            # time taken to measure self.dop candidates (per candidate) / we want to minimize this
            avg_meas_time_per_cand = (et - st) / self.task_states[str(self.task_idx)]["dop"]
            self.task_states[str(self.task_idx)]["times_so_far"].append(avg_meas_time_per_cand)

            if self.task_states[str(self.task_idx)]["dop"] > 1 and self.stage == "proper":
                
                # # done running, lets see if we need to rerun any
                part_results, cnt_fail_due_to_dop, cnt_err_tim_considered, succ_finally = self.remeasure_err_timeout(part_results, part_b_results, curr_configs)

                check = 0
                if succ_finally != 0:
                    check = (succ_finally / self.task_states[str(self.task_idx)]["dop"])

                # if not more than 25% of all failed candidates were due to dop
                if check > self.config["failed_due_to_dop_threshold"]:
                    remeas_indexes = self.get_remeasure_samples(part_results)

                    # re-measure chosen indexes serially:
                    logger.info("measurer re-measuring %d indexes: %s", len(remeas_indexes), str(remeas_indexes))
                    remeas_res = []
                    for idx in remeas_indexes:
                        rem_res, _ = self.run_configs([curr_configs[idx]], [part_b_results[idx]], remove_schedules=True if idx == remeas_indexes[-1] else False, run_type="remeasure",  set_timeout=self.determine_timeout(1))
                        remeas_res += rem_res

                    # calculate error for chosen indexes
                    avg_delta, deltas, avg_cmtp = self.calc_delta(part_results, remeas_res, remeas_indexes)

                    self.task_states[str(self.task_idx)]["deltas_so_far"] += deltas
                    # self.deltas_so_far += deltas

                    retval = self.update_results(remeas_indexes, remeas_res, part_results, avg_delta, avg_cmtp, theoretical_flop)
                    # retval = part_results
                else:
                    retval = part_results
                
                if cnt_fail_due_to_dop > int(self.task_states[str(self.task_idx)]["dop"] * self.config["failed_threshold"]):
                    perc_bad = (cnt_fail_due_to_dop / self.task_states[str(self.task_idx)]["dop"])

                self.task_states[str(self.task_idx)]["perc_bad_so_far"].append(perc_bad)
            else:
                retval = part_results
            
            all_results += retval

            if self.stage == "proper":
                self.optimizers_update(avg_delta, perc_bad, avg_meas_time_per_cand)

        m_inputs, m_results, conf_m_inputs, conf_m_results, total_error_count, error_counts = self.builder.orch_scheduler.get_inp_res_err(configs, all_results, theoretical_flop, task)
        Experiment.current.update_task_status(self.task_idx, len(configs), error_counts)
        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = False
        self.configurer.add_records(conf_m_inputs, conf_m_results)
        logger.info("measurer stop on #%s inputs, total errors: %s", str(len(configs)), str(sum(error_counts[1:])))
        return (m_inputs, m_results, total_error_count)
