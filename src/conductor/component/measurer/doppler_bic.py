from conductor._base import Configurable
from conductor.mediation import MeasureErrorNo
from conductor.component.measurer.doppler_base import DopplerBaseMeasurer
from conductor.experiment import Experiment
import logging
import tvm
import time
import numpy as np
import math

logger = logging.getLogger("conductor.component.measurer.doppler_bic")

class DopplerBicMeasurer(DopplerBaseMeasurer):
    _name = "doppler_bic"
    
    def __repr__(self):
        return DopplerBaseMeasurer.__repr__(self) + ":" + DopplerBicMeasurer._name

    def __init__(self, builder, runner, configs=None, child_default_configs={}):
        DopplerBaseMeasurer.__init__(self, "doppler_bic", builder, runner, configs=configs, child_default_configs=Configurable.merge_configs({
            "window_deltas": 2,
            "window_perc_good": 2,
            "window_times": 2,
            "adaptive_max_dop": 128,
            "beta": 0.2,
            "min_increment": 2,
            "max_increment": 7,
            "delta_threshold": 0.05,
            "failed_threshold": 0.05,
            "fixed_timeout": False,
        }, child_default_configs, override_first=True))

        self.window_deltas = self.config["window_deltas"]
        self.window_perc_good = self.config["window_perc_good"]
        self.window_times = self.config["window_times"]
        self.beta = self.config["beta"]
        self.min_increment = self.config["min_increment"]
        self.max_increment = self.config["max_increment"]

        self.task_state_template = {
            "deltas_so_far": [],
            "perc_good_so_far": [],
            "times_so_far": [],
            "last_it_failed": 0,
            "prior_max_dop": 0,
            "last_proposed_dop": 0,
            "dop": 1,
            "adaptive_max_dop": self.config["adaptive_max_dop"],
            "build_results": []
        }

        self.timings = {
            "building_cand": 0,
            "running_cand_initial": 0,
            "running_cand_err_rem": 0,
            "running_cand_rem": 0,
            "running_cand_rank": 0,
            "calc_dop": 0,
            "calc_analyze_err": 0,
            "calc_sampling": 0,
            "calc_deltas": 0,
            "calc_rank_sampling": 0
        }

    def print_list(self, lis, msg="", type="na"):
        for k, i in enumerate(lis):
            if type == "na":
                logger.warning("%s [%d]: %s", msg, k, str(i))
            elif type == "bld":
                logger.warning("%s [%d]: %s", msg, k, "er:[" + str(i.error_no) + "] " + i.filename)
            elif type == "msr":
                logger.warning("%s [%d]: %s", msg, k, "er:[" + str(i.error_no) + "] cst[" + str(i.costs) + "]")


    def speculative_analyze(self, mresults, inputs, configs, device_indices, cur_dop, theo_flop):
        #### TIME
        stcermtime = time.time()

        count_cands = len(mresults)
        unique_devices = list(set(device_indices))
        bydev = {ud: {
            "count": 0, "count_good_orig": 0, "count_err_orig": 0, "count_tim_err_orig": 0, "count_rtm_err_orig": 0,
            "count_good": 0, "count_err": 0, "count_tim_err": 0, "count_rtm_err": 0, 
            "count_err_dop": 0, "indices_good": [], "deltas": [], "cmtp": []
        } for ud in unique_devices}

        count_rem_err = 0
        count_successful = 1e20

        remeas_err_configs = []
        remeas_err_inputs = []
        remeas_err_indexes = []

        for idx, (inp, cfg, mres, devidx) in enumerate(zip(inputs, configs, mresults, device_indices)):
            if inp.error_no == MeasureErrorNo.NO_ERROR:
                if mres.error_no != MeasureErrorNo.NO_ERROR:
                    count_rem_err += 1
                    bydev[devidx]["count_err_orig"] += 1
                    remeas_err_configs.append(cfg)
                    remeas_err_inputs.append(inp)
                    remeas_err_indexes.append(idx)
                    if mres.error_no == MeasureErrorNo.RUN_TIMEOUT:
                        bydev[devidx]["count_tim_err_orig"] += 1
                    else:
                        bydev[devidx]["count_rtm_err_orig"] += 1
                else:
                    bydev[devidx]["count_good_orig"] += 1
                    bydev[devidx]["count_good"] += 1
                    bydev[devidx]["indices_good"].append(idx)
                    count_successful = min(bydev[devidx]["count_good_orig"], count_successful)
            bydev[devidx]["count"] += 1

        for devidx, val in bydev.items():
            logger.info(
                "DEV[%s](err_remeasure [before]): GOOD[%d/%d], ERR[%d/%d], E-TIM[%d/%d], E-RTM[%d/%d]",
                str(devidx) if not isinstance(devidx, str) else devidx, val["count_good_orig"], val["count"], val["count_err_orig"], val["count"],
                val["count_tim_err_orig"], val["count"], val["count_rtm_err_orig"], val["count"]
            )

        count_err_dop = 0

        #### TIME
        self.timings["calc_analyze_err"] += time.time() - stcermtime

        if count_rem_err > 0:
            #### TIME
            sermtime = time.time()
            remeas_err_results, _ = self.run_configs(
                remeas_err_configs, 
                remeas_err_inputs,
                self.spread_across(count_rem_err, self.device_ids),
                remove_schedules=False,
                run_type="error_remeasure",
                set_timeout=self.determine_timeout(count_rem_err // self.num_devs, num_devs=self.num_devs)
            )
            #### TIME
            self.timings["running_cand_err_rem"] += time.time() - sermtime

            #### TIME
            stcermtime = time.time()
            ret_results = [None] * count_cands
            for idx in range(count_cands):
                inp = inputs[idx]
                res = mresults[idx]
                devidx = device_indices[idx]
                if inp.error_no == MeasureErrorNo.NO_ERROR:
                    if res.error_no == MeasureErrorNo.NO_ERROR:
                        ret_results[idx] = res
                    else:
                        rem_res = remeas_err_results[remeas_err_indexes.index(idx)]
                        if res.error_no == MeasureErrorNo.RUN_TIMEOUT:
                            if rem_res.error_no == MeasureErrorNo.NO_ERROR:
                                bydev[devidx]["count_good"] += 1
                                bydev[devidx]["indices_good"].append(idx)
                                bydev[devidx]["count_err_dop"] += 1
                                count_err_dop = max(bydev[devidx]["count_err_dop"], count_err_dop)
                                ret_results[idx] = rem_res
                            else:
                                bydev[devidx]["count_err"] += 1
                                bydev[devidx]["count_tim_err"] += 1
                                ret_results[idx] = res
                        else:
                            if rem_res.error_no == MeasureErrorNo.NO_ERROR:
                                bydev[devidx]["count_good"] += 1
                                bydev[devidx]["indices_good"].append(idx)
                                bydev[devidx]["count_err_dop"] += 1
                                count_err_dop = max(bydev[devidx]["count_err_dop"], count_err_dop)
                                ret_results[idx] = rem_res
                            else:
                                bydev[devidx]["count_err"] += 1
                                bydev[devidx]["count_rtm_err"] += 1
                                ret_results[idx] = res
                else:
                    ret_results[idx] = res
                
            #### TIME
            self.timings["calc_analyze_err"] += time.time() - stcermtime
        else:
            ret_results = mresults
        
        for devidx, val in bydev.items():
            logger.info(
                "DEV[%s](err_remeasure [after]): GOOD[%d/%d], ERR[%d/%d], E-TIM[%d/%d], E-RTM[%d/%d], E-DOP[%d/%d]",
                str(devidx) if not isinstance(devidx, str) else devidx, val["count_good"], val["count"], val["count_err"], val["count"],
                val["count_tim_err"], val["count"], val["count_rtm_err"], val["count"],
                val["count_err_dop"], val["count"]
            )
        
        #### TIME
        check = 0
        remeasure_indexes = []
        for devidx, val in bydev.items():
            # check = max(0 if count_successful == 0 else (count_successful / cur_dop), check)
            # if check > self.config["failed_due_to_dop_threshold"]:
            stccalcrem = time.time()
            gi = val["indices_good"]
            cgi = len(gi)
            gprops = [self.builder.orch_scheduler.get_mean_cost(r) / r.total_time for k, r in enumerate(ret_results) if k in gi]
            if cgi > 2:
                if not self.config["fixed_samples"]:
                    outliers, _ = self.doubleMADsfromMedian(np.array(gprops), gi)
                    num_samples = min(max(len(outliers), math.ceil(self.config["perc_remeasure_samples"] * cgi)), cgi)
                    chosen_sampl_indexes = np.random.choice(gi, size=num_samples, replace=False).tolist()
                    if len(outliers) > 0:
                        set_indexes = set(outliers)
                        while len(set_indexes) < num_samples:
                            set_indexes.add(chosen_sampl_indexes.pop(0))
                    else:
                        set_indexes = set(chosen_sampl_indexes)
                    remeasure_indexes += list(set_indexes)
                    logger.info(
                        "measurer: (sampling) DEV[%s]:: found %d outliers: %s, picking %d samples to remeasure: %s", 
                        str(devidx) if not isinstance(devidx, str) else devidx, len(outliers), str(outliers), len(list(set_indexes)), str(list(set_indexes))
                    )
                else:
                    if self.config["fixed_sample_count"] > 0:
                        remeasure_indexes = np.random.choice(gi, size=self.config["fixed_sample_count"], replace=False).tolist()
                        logger.info(
                            "measurer: (sampling) DEV[%s]:: picking %d samples to remeasure: %s", 
                            str(devidx) if not isinstance(devidx, str) else devidx, len(remeasure_indexes), str(remeasure_indexes)
                        )

            #### TIME
            self.timings["calc_sampling"] += time.time() - stccalcrem

        deltas = []
        avg_delta = 0

        #### TIME
        strcrem = time.time()

        if len(remeasure_indexes) > 0:
            remeasure_results, _ = self.run_configs(
                [configs[idx] for idx in remeasure_indexes],
                [inputs[idx] for idx in remeasure_indexes],
                self.spread_across(len(remeasure_indexes), self.device_ids),
                n_parallel=1,
                remove_schedules=False,
                run_type="remeasure",
                set_timeout=self.determine_timeout(1, num_devs=self.num_devs)
            )
        else:
            remeasure_results = []

        #### TIME
        self.timings["running_cand_rem"] += time.time() - strcrem  

        count_rem_good = 0

        # stats = []

        #### TIME
        stdeltas = time.time()
        for idx in range(len(ret_results)):
            res = ret_results[idx]
            devidx = device_indices[idx]
            if idx in remeasure_indexes:
                rem_res = remeasure_results[remeasure_indexes.index(idx)]
                if rem_res.error_no == MeasureErrorNo.NO_ERROR and res.error_no == MeasureErrorNo.NO_ERROR:
                    rem_perf = self.builder.orch_scheduler.get_mean_cost(rem_res)
                    orig_perf = self.builder.orch_scheduler.get_mean_cost(res)
                    delt = abs(orig_perf - rem_perf) / orig_perf
                    bydev[devidx]["deltas"].append(delt)
                    if rem_perf < orig_perf:
                        ret_results[idx] = rem_res
                    count_rem_good += 1

        if count_rem_good >= 2:
            for devidx in bydev:
                if np.sum(deltas) < np.sum(bydev[devidx]["deltas"]):
                    deltas = bydev[devidx]["deltas"]
                nmd = np.mean(bydev[devidx]["deltas"])
                if avg_delta < nmd:
                    avg_delta = nmd
                    

                bydev[devidx]["cmtp"] = np.mean(bydev[devidx]["cmtp"])
                
                logger.info(
                    "measurer: (calc deltas) DEV[%s]:: AVGDelta: %f from #%d deltas",
                    str(devidx) if not isinstance(devidx, str) else devidx,
                    np.mean(bydev[devidx]["deltas"]), 
                    len(bydev[devidx]["deltas"])
                )
                bydev[devidx]["deltas"] = nmd

            for idx in range(len(ret_results)):
                res = ret_results[idx]
                devidx = device_indices[idx]
                if idx in bydev[devidx]["indices_good"] and idx not in remeasure_indexes and bydev[devidx]["deltas"] < 0.25:
                    perf = self.builder.orch_scheduler.get_mean_cost(res)
                    ttime = res.total_time
                    prop = perf / ttime
                    ntt = ttime - (ttime * bydev[devidx]["deltas"])
                    nprop = prop
                    nperf = nprop * ntt
                    ret_results[idx] = self.builder.orch_scheduler.get_measure_result(
                        [nperf], res.error_no, res.error_msg, res.all_cost, res.timestamp, 
                        theo_flop / nperf if nperf != 0 else 0, nperf, res.total_time,
                        "*" if res.error_no == MeasureErrorNo.NO_ERROR else "E"
                    )

        #### TIME
        self.timings["calc_deltas"] += time.time() - stdeltas

        return (ret_results, deltas, avg_delta, count_err_dop)


    def prepare_dop(self, all_inp_configs):
        prop_dop = self.task_states[str(self.task_idx)]["dop"]
        lc = len(all_inp_configs)
        if self.stage == "proper":
            # dop must be at least 1
            tmp_dop = max(1, prop_dop)

            # we have enough to satisfy all devices with dop
            if self.num_devs * tmp_dop <= lc:
                # take num_devs * dop configs from the front
                to_take = self.num_devs * tmp_dop
                ret_configs, curr_configs, curr_bld_res = self.pick_n_configs(all_inp_configs, to_take)
                device_indices = self.spread_across(to_take, self.device_ids)
                self.task_states[str(self.task_idx)]["dop"] = tmp_dop
                self.task_states[str(self.task_idx)]["last_proposed_dop"] = tmp_dop
            else:
                ret_configs, curr_configs, curr_bld_res = self.pick_n_configs(all_inp_configs, lc)
                device_indices = self.spread_across(lc, self.device_ids)
                self.task_states[str(self.task_idx)]["dop"] = 1 if lc < self.num_devs else math.ceil(lc / self.num_devs)
                self.task_states[str(self.task_idx)]["last_proposed_dop"] = 1 if lc < self.num_devs else math.ceil(lc / self.num_devs)  
        else:
            to_take = min(self.num_devs, lc)
            ret_configs, curr_configs, curr_bld_res = self.pick_n_configs(all_inp_configs, to_take)
            device_indices = self.device_ids[:to_take]
            self.task_states[str(self.task_idx)]["dop"] = 1
            self.task_states[str(self.task_idx)]["last_proposed_dop"] = 1
            tmp_dop = 1

        cur_dop = self.task_states[str(self.task_idx)]["dop"]

        logger.info("measurer proposed_dop[%d], actual cur_dop[%d], configs[%d], configs_left[%d]", prop_dop, cur_dop, lc, len(ret_configs))
        return (ret_configs, curr_configs, curr_bld_res, device_indices, cur_dop)

    def split_across_temp(self, arr_len, dev_ids):
        dev_idx = 0
        split_list = []
        num_devs = len(dev_ids)
        for _ in range(arr_len):
            split_list.append(dev_ids[dev_idx])
            if dev_idx < num_devs - 1:
                dev_idx += 1
            else:
                dev_idx = 0
        return split_list


    def measure(self, task, configs, options=None):
        Experiment.current.set_experiment_stage("measurer:measure")
        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = True
        logger.info("measurer start on #%s inputs", str(len(configs)))
        
        self.create_task_state(self.task_idx)

        sbtime = time.time()
        # at this point, we have all build inputs prepared.
        dev_map = self.split_across_temp(len(configs), self.device_ids)
        build_inputs, updated_configs, theoretical_flop = self.builder.orch_scheduler.get_build_inputs(
            task,
            configs,
            dev_map,
            self.runner.dev_ctx_details,
            self.hash_callback,
            options=options
        )

        bld_results = self.build_configs(build_inputs)
        all_inp_configs = list(zip(bld_results, updated_configs))
        all_results = []

        self.task_states[str(self.task_idx)]["build_results"] += bld_results

        self.timings["building_cand"] += time.time() - sbtime
        while len(all_inp_configs) > 0:
            scdptime = time.time()
            all_inp_configs, curr_configs, curr_bld_res, device_indices, cur_dop = self.prepare_dop(all_inp_configs)

            self.timings["calc_dop"] += time.time() - scdptime
            
            srcitime = time.time()
            st = time.time()
            # run them
            part_results, part_b_results = self.run_configs(curr_configs, curr_bld_res, device_indices, n_parallel=None, remove_schedules=False, run_type="regular", set_timeout=self.determine_timeout(cur_dop, num_devs=len(set(device_indices))))
            et = time.time()
            self.timings["running_cand_initial"] += time.time() - srcitime
            
            # initially, error is 0
            avg_delta = 0
            deltas = []

            # percentage of bad (not errored candidates is initially 0.0)
            perc_good = 1.0

            count_err_dop = 0
            
            # time taken to measure self.dop candidates (per candidate) / we want to minimize this
            avg_meas_time_per_cand = (et - st) / cur_dop
            self.task_states[str(self.task_idx)]["times_so_far"].append(avg_meas_time_per_cand)

            if cur_dop > 1 and self.stage == "proper":
                retval, deltas, avg_delta, count_err_dop = self.speculative_analyze(
                    part_results, part_b_results, curr_configs, device_indices, cur_dop, theoretical_flop
                )
                perc_good = 1.0 if count_err_dop <= 0 else (count_err_dop / cur_dop)
                self.task_states[str(self.task_idx)]["perc_good_so_far"].append(perc_good)
                self.task_states[str(self.task_idx)]["deltas_so_far"] += deltas
                logger.info("measurer: (done) perc_good: %f, avg_delta: %f, error_dop: %d", perc_good, avg_delta, count_err_dop)



            else:
                retval = part_results

            all_results += retval

            if self.stage == "proper":
                scdptime = time.time()

                self.task_states[str(self.task_idx)]["dop"] += 1
                if avg_delta > self.config["delta_threshold"] or count_err_dop > int(cur_dop * self.config["failed_threshold"]):
                    logger.info("Errored too high due to parallelism or delta error too high.")
                    if self.task_states[str(self.task_idx)]["dop"] < self.task_states[str(self.task_idx)]["adaptive_max_dop"]:
                        # fast convergence --- this should happen when dop was tuned to be huge
                        # and bigger than our 'max dop'
                        self.task_states[str(self.task_idx)]["adaptive_max_dop"] = int(self.task_states[str(self.task_idx)]["dop"] * (2 - self.beta) / 2)
                    else:
                        self.task_states[str(self.task_idx)]["adaptive_max_dop"] = int(self.task_states[str(self.task_idx)]["dop"])
                    
                    self.task_states[str(self.task_idx)]["dop"] = math.floor(self.task_states[str(self.task_idx)]["dop"] * (1 - self.beta))
                    logger.info("Dropped Adaptive max dop to [%d] and dop to [%d]", self.task_states[str(self.task_idx)]["adaptive_max_dop"], self.task_states[str(self.task_idx)]["dop"])
                    continue
                
                bic_increment = 0
                if self.task_states[str(self.task_idx)]["dop"] < self.task_states[str(self.task_idx)]["adaptive_max_dop"]:
                    bic_increment = (self.task_states[str(self.task_idx)]["adaptive_max_dop"] - self.task_states[str(self.task_idx)]["dop"]) / 2
                else:
                    bic_increment = self.task_states[str(self.task_idx)]["dop"] - self.task_states[str(self.task_idx)]["adaptive_max_dop"]
                
                if bic_increment > self.max_increment:
                    bic_increment = self.max_increment
                elif bic_increment < self.min_increment:
                    bic_increment = self.min_increment
                
                old = self.task_states[str(self.task_idx)]["dop"]
                self.task_states[str(self.task_idx)]["dop"] = math.floor(self.task_states[str(self.task_idx)]["dop"] + bic_increment)

                avg_delta_win = self.get_window_average(self.task_states[str(self.task_idx)]["deltas_so_far"], self.window_deltas)
                avg_perc_good_win = self.get_window_average(self.task_states[str(self.task_idx)]["perc_good_so_far"], self.window_perc_good)
                avg_time_win = self.get_window_average(self.task_states[str(self.task_idx)]["times_so_far"], self.window_times)

                logger.info("TASK[%s] [ TIME  ] current(%f), avg(%f)", str(self.task_idx), avg_meas_time_per_cand, avg_time_win)
                logger.info("TASK[%s] [ DELTA ] current(%f), avg(%f)", str(self.task_idx), avg_delta_win, avg_delta)
                logger.info("TASK[%s] [ GOOD  ] current(%f), avg(%f)", str(self.task_idx), count_err_dop, avg_perc_good_win)
                logger.info("Increasing dop to [%d] from [%d]", self.task_states[str(self.task_idx)]["dop"], old)

                self.timings["calc_dop"] += time.time() - scdptime

        m_inputs, m_results, conf_m_inputs, conf_m_results, total_error_count, error_counts = self.builder.orch_scheduler.get_inp_res_err(configs, all_results, theoretical_flop, task)
        Experiment.current.update_task_status(self.task_idx, len(configs), error_counts)
        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = False
        self.configurer.add_records(conf_m_inputs, conf_m_results)
        logger.info("measurer stop on #%s inputs, total errors: %s", str(len(configs)), str(sum(error_counts[1:])))
        return (m_inputs, m_results, total_error_count)

