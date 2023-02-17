#!/bin/bash

interp=$(which python3)

sudo CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    ${interp} -m conductor.component.rpc.default.run_server --key $1 --gpu_idx $2 --tracker_host $3 --host $4
