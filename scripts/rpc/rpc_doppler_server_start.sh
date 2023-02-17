#!/bin/bash

interp=$(which python3)

sudo CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    ${interp} -m conductor.component.rpc.doppler.server.main --name $1 --device $2 --dev_ids $3 --tracker_host $4 --host $5
