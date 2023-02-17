from __future__ import absolute_import
import sys
import logging
import signal
from tvm import rpc
import os
import logging
logger = logging.getLogger("conductor.component.rpc.server")


def main(tracker, key, host, port, port_end, gpu_idx):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    if tracker:
        url, port = tracker.rsplit(":", 1)
        port = int(port)
        tracker_addr = (url, port)
        if not key:
            raise RuntimeError("Need key to present type of resource when tracker is available")
    else:
        tracker_addr = None

    server = rpc.Server(
        host,
        port,
        port_end,
        key=key,
        tracker_addr=tracker_addr,
        load_library=None,
        custom_addr=None,
        silent=False,
    )
    
    def handler(sig, frame):
        server.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    server.proc.join()