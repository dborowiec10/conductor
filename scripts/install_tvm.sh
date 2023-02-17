#!/bin/bash
interp=$(which python3)

pushd tvm-conductor
    pushd build
        if [ ! -f build.ninja ]
        then
            cmake .. -G Ninja
        fi
        ninja
    popd
    pushd python
        ${interp} setup.py install &> /dev/null 
    popd
popd