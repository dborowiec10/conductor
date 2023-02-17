# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Multiprocessing via Popen.

This module provides a multi-processing pool backed by Popen.
with additional timeout support.
"""
import os
import sys
import struct
import subprocess
import threading
from enum import IntEnum
import pickle
import signal
import time
import cloudpickle
import select

class StatusKind(IntEnum):
    RUNNING = 0
    COMPLETE = 1
    EXCEPTION = 2
    TIMEOUT = 3

def kill_child_processes(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in children:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass

def check_exec(reader, timeout, callback, killer, pass_back):
    start_time = time.time()
    timed_out = True
    status = StatusKind.TIMEOUT
    value = None
    err_msg = str(TimeoutError())
    len_data = 0
    rfd, pll = reader
    
    while timed_out and (time.time() - start_time) < timeout:
        # poll file descriptor for data
        poll_result = pll.poll(0)

        # if we have some data
        if poll_result:
            # no longer considering a worker-process-external timeout
            timed_out = False
            try:
                # attempt to read data from descriptor
                len_data = rfd.read(4)
                if len(len_data) != 0:
                    # we have meaningful output
                    status, value = cloudpickle.loads(rfd.read(struct.unpack("<i", len_data)[0]))
                    err_msg = ""
                else:
                    status = StatusKind.EXCEPTION
                    err_msg = str(ChildProcessError("Subprocess terminated"))
                    killer()

            except IOError as e:
                status = StatusKind.EXCEPTION
                err_msg = str(ChildProcessError("Subprocess terminated"))
                killer()
        
        else:
            time.sleep(0.5)
    
    retval = (status, err_msg, value)
    if callback is not None:
        callback(retval)
    pass_back[0] = retval

class ExecutingWorker():
    _name = "executing_worker"

    def __repr__(self):
        return ExecutingWorker._name

    def __init__(self, start=True):
        self._proc = None
        self.timeout = None
        self.last_start_time = None
        self.check_thread = None
        self.pass_back = [None]
        self.status = False
        if start:
            self._start()
        
    @staticmethod
    def kill_all(workers):
        for w in workers:
            w.kill()

    def __del__(self):
        try:
            self.kill()
        except ImportError:
            pass

    def kill(self):
        if self._proc is not None:
            try:
                self._writer.close()
            except IOError:
                pass
            try:
                self._reader[0].close()
            except IOError:
                pass
            try:
                kill_child_processes(self._proc.pid)
            except TypeError:
                pass
            try:
                self._proc.kill()
            except OSError:
                pass
            self._proc = None
            self.status = False

    def start(self):
        self._start()
    
    def _start(self):
        if self._proc is not None:
            return

        main_read, worker_write = os.pipe()
        worker_read, main_write = os.pipe()

        cmd = [sys.executable, "-m", "conductor.worker.worker_process"]
        cmd += [str(worker_read), str(worker_write)]
        self._proc = subprocess.Popen(cmd, pass_fds=(worker_read, worker_write))
        os.close(worker_read)
        os.close(worker_write)
        rfd = os.fdopen(main_read, "rb")
        pll = select.poll()
        pll.register(rfd, select.POLLIN)
        self._reader = (rfd, pll)
        self._writer = os.fdopen(main_write, "wb")
        self.status = True

    def submit(self, fn, args=(), timeout=None, callback=None):
        import cloudpickle

        if self._proc is None:
            self._start()

        data = cloudpickle.dumps((fn, args), protocol=pickle.HIGHEST_PROTOCOL)

        try:
            self._writer.write(struct.pack("<i", len(data)))
            self._writer.write(data)
            self._writer.flush()
            
        except IOError:
            pass

        self.timeout = timeout
        self.last_start_time = time.time()
        # by this point, the process should be started and received a message with fn/args

        # start a checker_thread here
        self.check_thread = threading.Thread(
            target=check_exec, 
            args=(
                self._reader, 
                self.timeout, 
                callback, 
                self.kill,
                self.pass_back
            )
        )
        self.check_thread.start()

    def get(self):
        self.check_thread.join()
        return self.pass_back[0]

        



