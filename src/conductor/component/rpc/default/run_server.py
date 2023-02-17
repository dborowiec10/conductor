from __future__ import absolute_import, print_function

from absl import app
from absl import flags

import os

import multiprocessing as _multi
from conductor.component.rpc.default.server import main as main_server
from conductor.component.rpc.default.utils import find_tracker_port

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "0.0.0.0", "host ip address for server/tracker")
flags.DEFINE_string("tracker_host", "0.0.0.0", "host ip address for the tracker")
flags.DEFINE_integer("port", 9000, "host port for server/tracker")
flags.DEFINE_integer("port_end", 9199, "host end port for server/tracker")
flags.DEFINE_string("key", "localdevice", "name of device key")
flags.DEFINE_integer("gpu_idx", 0, "idx of the gpu to use")

def main(argv):
    del argv
    print("RUNNING SERVER", FLAGS.gpu_idx)
    tracker_port = find_tracker_port(FLAGS.tracker_host, FLAGS.port, FLAGS.port_end)
    multi = _multi.get_context("spawn")
    p = multi.Process(target=main_server, args=(FLAGS.tracker_host + ":" + str(tracker_port), FLAGS.key, FLAGS.host, FLAGS.port, FLAGS.port_end, FLAGS.gpu_idx))
    p.start()


if __name__ == '__main__':
    app.run(main)