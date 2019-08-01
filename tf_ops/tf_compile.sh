#!/usr/bin/env bash

cd ./sampling
./tf_sampling_compile.sh
cd ../
echo "compile sampling finished!"

cd ./grouping
./tf_grouping_compile.sh
cd ../
echo "compile grouping finished!"

cd ./nms_3d
./tf_nms3d_compile.sh
cd ../
echo "compile nms finished!"

cd ./interpolation_3d
./tf_interpolate_compile.sh
cd ../
echo "compile 3d_interpolate finished!"
