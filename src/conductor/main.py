from __future__ import absolute_import, print_function

from absl import app
from absl import flags

import os
import logging
import json
import glob
import datetime
import traceback

from conductor.job._base import JobSpecification
from conductor._base import get_conductor_path

from pynvml import *

FLAGS = flags.FLAGS
flags.DEFINE_string("jobs_path", "jobs/current", "path to the job specification file directory")
flags.DEFINE_string("spec_path", "empty", "path to the job specification file")
flags.DEFINE_string("results_path", "./data/results/", "path to a directory where to store job results")
flags.DEFINE_string("models_path", None, "path to a directory where model images are stored")
flags.DEFINE_string("tensor_programs_path", None, "path to a directory where operator/subgraph schedulables are stored in pickled form")
flags.DEFINE_string("mongo_host", "100.127.131.25", "hostname of mongodb instance")
flags.DEFINE_integer("mongo_port", 27017, "port of mongodb instance")
flags.DEFINE_string("mongo_user", "doppler", "username for mongodb instance")
flags.DEFINE_string("mongo_pass", "doppler", "password for mongodb instance")
flags.DEFINE_string("mongo_db", "doppler_remote", "mongodb database")

def setup_logging(res_path, job_name):
    fileh = logging.FileHandler(os.path.join(res_path, "run.log"), 'a')
    formatter = logging.Formatter('%(asctime)s - ' + job_name + ' - [%(name)s] - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)      # set the new handler

def load_spec(path):
    with open(path, "r") as spec_file:
        spec = JobSpecification(json.load(spec_file))
    return spec

def setup_custom_logger(name):
    logging.getLogger('autotvm').setLevel(logging.WARNING)
    logging.getLogger('compile_engine').setLevel(logging.WARNING)
    logging.getLogger('auto_scheduler').setLevel(logging.WARNING)
    logging.getLogger('TVMC').setLevel(logging.WARNING)
    logging.getLogger('Common').setLevel(logging.WARNING)
    logging.getLogger('TensorRT').setLevel(logging.WARNING)
    logging.getLogger('strategy').setLevel(logging.WARNING)
    logging.getLogger('RPCServer').setLevel(logging.WARNING)
    logging.getLogger('RPCTracker').setLevel(logging.WARNING)
    logging.getLogger('topi').setLevel(logging.WARNING)
    logging.getLogger('conv2d_winograd').setLevel(logging.WARNING)
    logging.getLogger('conv3d_winograd').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('apscheduler.executors').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    logging.getLogger('apscheduler.jobstores').setLevel(logging.WARNING)
    logging.getLogger('concurrent.futures').setLevel(logging.WARNING)
    logging.getLogger('concurrent').setLevel(logging.WARNING)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # disable weird tensorflow messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def conductor_job(job_spec, results_path, models_path, tensor_programs_path):
    job_name = str(datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')) + "_" + job_spec.name
    job_results_path = os.path.join(results_path, job_name)
    os.makedirs(job_results_path, exist_ok=True)
    setup_logging(job_results_path, job_name)
    logger = logging.getLogger("conductor.main")

    preamble = "START EXECUTING JOB: " + job_spec.name
    logger.info(preamble)

    try:
        job = job_spec.from_spec(job_results_path, models_path, tensor_programs_path, {
            "host": FLAGS.mongo_host,
            "port": FLAGS.mongo_port,
            "user": FLAGS.mongo_user,
            "pass": FLAGS.mongo_pass,
            "db": FLAGS.mongo_db
        })
        job.run()

    except Exception as e:
        failamble = "JOB FAILED: " + job_spec.name
        logger.error(failamble)
        s = traceback.format_exc()
        logger.error(s)

    postamble = "FINISH EXECUTING JOB: " + job_spec.name
    logger.info(postamble)

def conductor_single_job(path, results_path, models_path, tensor_programs_path):
    conductor_job(load_spec(path), results_path, models_path, tensor_programs_path)

def conductor_multi_job(path, results_path, models_path, tensor_programs_path):
    filelist = glob.glob(os.path.join(path, '*.json'))
    for infile in sorted(filelist): 
        conductor_job(load_spec(infile), results_path, models_path, tensor_programs_path)

def run_conductor(results_path, models_path, tensor_programs_path):
    if FLAGS.spec_path != "empty":
        conductor_single_job(FLAGS.spec_path, results_path, models_path, tensor_programs_path)
    else:
        conductor_multi_job(FLAGS.jobs_path, results_path, models_path, tensor_programs_path)

def main(argv):
    del argv
    setup_custom_logger("conductor")
    
    if FLAGS.models_path is not None:
        models_path = os.path.abspath(FLAGS.models_path)
        os.makedirs(models_path, exist_ok=True)
    else:
        models_path = None

    if FLAGS.tensor_programs_path is not None:
        tensor_programs_path = os.path.abspath(FLAGS.tensor_programs_path)
        os.makedirs(tensor_programs_path, exist_ok=True)
    else:
        tensor_programs_path = os.path.join(get_conductor_path(), "tensor_programs")

    os.makedirs(FLAGS.results_path, exist_ok=True)
    run_conductor(FLAGS.results_path, models_path, tensor_programs_path)


if __name__ == '__main__':
    app.run(main)