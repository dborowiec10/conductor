from __future__ import absolute_import, print_function

from absl import app
from absl import flags

import multiprocessing as _multi
from conductor.component.rpc.default.server import main as main_server
from conductor.component.rpc.default.tracker import main as main_tracker

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "0.0.0.0", "host ip address for server/tracker")
flags.DEFINE_integer("port", 9000, "host port for server/tracker")
flags.DEFINE_integer("port_end", 9199, "host end port for server/tracker")

def main(argv):
    del argv
    
    tracker_port = -1

    multi = _multi.get_context("spawn")
    queue = multi.Queue()
    p = multi.Process(target=main_tracker, args=(FLAGS.host, FLAGS.port, FLAGS.port_end, False, queue))
    p.start()
    while (True):
        try:
            tracker_port = queue.get(timeout=3)
            break
        except Exception as error:
            print(error)
            break


if __name__ == '__main__':
    app.run(main)