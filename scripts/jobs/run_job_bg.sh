#!/bin/bash

interp=$(which python3)

conductor_path=$HOME/.conductor
models_path=$HOME/.conductor/models
tensor_programs_path=$HOME/.conductor/tensor_programs

sudo CONDUCTOR_PATH=$conductor_path \
    CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    nohup ${interp} src/conductor/main.py \
        --spec_path $1 \
        --models_path $models_path \
        --tensor_programs_path $tensor_programs_path &
