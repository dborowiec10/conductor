#!/bin/bash

pushd src
    interp=$(which python3)
    ${interp} setup.py clean --all install clean --all
popd