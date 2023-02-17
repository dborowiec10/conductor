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
"""Internal PopenWorker for PopenPool."""
import sys
import os
import struct
import traceback
import pickle
import logging
import cloudpickle

from conductor.worker.worker import StatusKind

def main():
    if len(sys.argv) != 3:
        print("Usage: <read_fd> <write_fd>")
        return

    reader = os.fdopen(int(sys.argv[1]), "rb")
    writer = os.fdopen(int(sys.argv[2]), "wb")

    logging.basicConfig(level=logging.INFO)

    def _respond(ret_value):
        data = cloudpickle.dumps(ret_value, protocol=pickle.HIGHEST_PROTOCOL)
        writer.write(struct.pack("<i", len(data)))
        writer.write(data)
        writer.flush()

    while True:
        raw_bytes_size = reader.read(4)
        if len(raw_bytes_size) != 4:
            return
        bytes_size = struct.unpack("<i", raw_bytes_size)[0]
        fn, args = cloudpickle.loads(reader.read(bytes_size))
        try:
            result = fn(args)
            ret_value = (StatusKind.COMPLETE, result)
        except Exception as exception:
            msg = traceback.format_exc()
            ret_value = (StatusKind.EXCEPTION, type(exception)(msg))
        
        _respond(ret_value)
        

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, IOError):
        pass
