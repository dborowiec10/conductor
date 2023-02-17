#!/bin/bash

interp=$(which python3)

sudo CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    ${interp} -m conductor.component.rpc.default.run_tracker --host $1 --port $2
