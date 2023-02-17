from __future__ import absolute_import, print_function

from absl import app
from absl import flags


import logging
import os
import rpyc
from pprint import pprint

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "0.0.0.0", "host ip address for tracker")
flags.DEFINE_integer("port", 9000, "host port for tracker")

def connect_tracker(tracker_host, tracker_port):
    tc = rpyc.connect(tracker_host, tracker_port)
    try:
        ret = tc.root.ping()
        if ret != "pong":
            raise Exception("Unable to connect to tracker")
    except Exception as e:
        raise Exception("Unable to connect to tracker")
    return tc

def main(argv):
    del argv
    logging.basicConfig(level=logging.INFO)
    tc = connect_tracker(FLAGS.host, FLAGS.port)
    pprint(tc.root.get_servers())


    # tracker_port = find_tracker_port(FLAGS.host, FLAGS.port, FLAGS.port_end)

    # conn = rpc.connect_tracker(FLAGS.host, tracker_port)

    # print("Tracker address %s:%d\n" % (FLAGS.host, tracker_port))
    # print("%s" % conn.text_summary())

if __name__ == '__main__':
    app.run(main)
