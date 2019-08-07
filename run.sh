#!/usr/bin/env bash

cd tf_ops
./tf_compile.sh
cd ../

CUDA_VISIBLE_DEVICES=0,1 python3 run.py
