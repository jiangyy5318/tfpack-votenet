#!/usr/bin/env bash

cd tf_ops
./tf_compile.sh
cd ../
find . -name '*__pycache__' | xargs rm -rf
rm -rf train_log/${1}
CUDA_VISIBLE_DEVICES=0,1 python3 ${1}.py
