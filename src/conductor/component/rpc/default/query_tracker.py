from __future__ import absolute_import, print_function

from absl import app
from absl import flags


import logging
import os
from tvm import rpc

from conductor.component.rpc.default.utils import find_tracker_port

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "0.0.0.0", "host ip address for tracker")
flags.DEFINE_integer("port", 9000, "host port for tracker")
flags.DEFINE_integer("port_end", 9199, "host end port for tracker")

def main(argv):
    del argv
    logging.basicConfig(level=logging.INFO)

    tracker_port = find_tracker_port(FLAGS.host, FLAGS.port, FLAGS.port_end)

    conn = rpc.connect_tracker(FLAGS.host, tracker_port)
    # pylint: disable=superfluous-parens
    print("Tracker address %s:%d\n" % (FLAGS.host, tracker_port))
    print("%s" % conn.text_summary())

if __name__ == '__main__':
    app.run(main)
