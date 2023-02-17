#!/bin/bash

interp=$(which python3)
nsyss=$(which nsys)
hostname=$(hostname)

j1="manual_jobs/current/$1/00_batch_matmul_51268830.cuda_standalone_template_grid_index_default_nvtx_cuda_sched.yaml"
j2="manual_jobs/current/$1/06_conv1d_ncw_f691e8dc.cuda_standalone_template_grid_index_default_nvtx_cuda_sched.yaml"
j3="manual_jobs/current/$1/12_conv2d_hwcn_f0d007d2.cuda_standalone_template_grid_index_default_nvtx_cuda_sched.yaml"
j4="manual_jobs/current/$1/28_conv3d_ncdhw_6ef0f2a0.cuda_standalone_template_grid_index_default_nvtx_cuda_sched.yaml"

sudo CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    $nsyss profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        -o tvm_profs/tvm_timeline.$hostname.job_00.parallel_${1}_${2}.%p.prof \
            ${interp} src/conductor/core/main.py --spec_path $j1
            
sudo CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    $nsyss profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        -o tvm_profs/tvm_timeline.$hostname.job_06.parallel_${1}_${2}.%p.prof \
            ${interp} src/conductor/core/main.py --spec_path $j2

sudo CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    $nsyss profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        -o tvm_profs/tvm_timeline.$hostname.job_12.parallel_${1}_${2}.%p.prof \
            ${interp} src/conductor/core/main.py --spec_path $j3

sudo CPATH=$CPATH \
    PATH=$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    TVM_HOME=$TVM_HOME \
    TVM_ROOT=$TVM_ROOT \
    CUDA_PATH=$CUDA_PATH \
    $nsyss profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        -o tvm_profs/tvm_timeline.$hostname.job_28.parallel_${1}_${2}.%p.prof \
            ${interp} src/conductor/core/main.py --spec_path $j4



