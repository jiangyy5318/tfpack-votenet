#!/usr/bin/env bash

cd tf_logs
./tf_compile.sh
cd ../

log=nohup.out
rm -rf ${log}
CUDA_VISIBLE_DEVICES=0,1 nohup python3 run.py &> ${log}&