#!/bin/bash
interp=$(which python3) || :
kill -9 $(pgrep -f ${interp}) || :
kill -9 $(pgrep -f tvm.exec.rpc_server) || :
kill -9 $(pgrep -f nvidia-cuda-mps-control) || :
kill -9 $(pgrep -f nvidia-cuda-mps-server) || : 
kill -9 $(pgrep -f conductor) || :
kill -9 $(pgrep -f tvm) || :
echo "Killing MPS"
nvidia-smi -c DEFAULT