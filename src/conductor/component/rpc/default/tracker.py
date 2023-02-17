from __future__ import absolute_import

import logging
import sys
import signal
from tvm.rpc.tracker import Tracker

import logging
logger = logging.getLogger("conductor.component.rpc.tracker")

def signal_handler(tracker):
    def handler(sig, frame):
        tracker._stop_tracker()
        tracker.terminate()
        sys.exit(0)
    return handler

def main(host, port, port_end, silent, queue):
    tracker = Tracker(host, port=port, port_end=port_end, silent=silent)
    signal.signal(signal.SIGINT, signal_handler(tracker))
    queue.put(tracker.port)
    tracker.proc.join()