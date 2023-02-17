import tvm

from conductor._base import Configurable
from conductor.component.measurer._base import Measurer
from conductor.experiment import Experiment

import logging
logger = logging.getLogger("conductor.component.measurer.default")

class DefaultMeasurer(Measurer):
    _name = "default"
    
    def __repr__(self):
        return Measurer.__repr__(self) + ":" + DefaultMeasurer._name

    def __init__(self, builder, runner, configs=None):
        Measurer.__init__(self,
            "default", 
            builder, 
            runner, 
            configs=configs, 
            child_default_configs=Configurable.merge_configs({}, {}, override_first=True)
        )

    def split_across(self, arr_len, dev_ids):
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
        logger.info("measurer start on #%s inputs", str(len(configs)))
        
        # here we need to map each candidate to a device
        dev_map = self.split_across(len(configs), self.device_ids)

        build_inputs, updated_configs, theoretical_flop = self.builder.orch_scheduler.get_build_inputs(
            task,
            configs,
            dev_map,
            self.runner.dev_ctx_details,
            self.hash_callback,
            options=options
        )

        build_results = self.builder.build(build_inputs)
        results = self.runner.run(updated_configs, build_results, dev_map)
        Experiment.current.set_experiment_stage("measurer:measure")

        m_inputs, m_results, conf_m_inputs, conf_m_results, total_error_count, error_counts = self.builder.orch_scheduler.get_inp_res_err(configs, results, theoretical_flop, task)
        self.configurer.add_records(conf_m_inputs, conf_m_results)
        Experiment.current.update_task_status(self.task_idx, len(configs), error_counts)
        logger.info("measurer stop on #%s inputs, total errors: %s", str(len(configs)), str(sum(error_counts[1:])))
        return (m_inputs, m_results, total_error_count)
