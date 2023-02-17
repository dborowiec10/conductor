import logging
from unittest import result
from conductor.experiment import experiments
logger_job = logging.getLogger("conductor.job")

class Job(object):
    _name = "job"

    def __repr__(self):
        return Job._name

    def __init__(self, spec, job_results_path, models_path, tensor_programs_path, results_db_spec):
        self.results_db_spec = results_db_spec
        self.spec = spec
        self.name = self.spec.name
        self.inputs = {}
        for isp in self.spec.inputs:
            pth = models_path if isp._type in ["model", "compiled_model"] else tensor_programs_path
            self.inputs[isp.name] = isp.from_spec(pth)
        self.task = self.spec.task.from_spec(job_results_path, models_path, tensor_programs_path)
        self.task_inputs = [(i, self.inputs[i]) for i in self.spec.task.inputs]
        logger_job.info("loaded task")

    def run(self):
        with experiments[self.task._type + ":" + self.task.subtype](self.results_db_spec, self.name, self.spec):
            logger_job.info("running job %s", self.name)
            logger_job.info("start executing task: %s", self.task.name)
            self.task.load()
            self.task.run(self.task_inputs, 0)
            logger_job.info("stop executing task: %s", self.task.name)


