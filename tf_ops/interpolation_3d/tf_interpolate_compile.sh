#/bin/bash
#!/usr/bin/env bash

NVCC=/usr/local/cuda/bin/nvcc
#TF_INCLUDE=/usr/local/lib/python3.6/dist-packages/tensorflow/include/
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=/usr/local/lib/python3.6/dist-packages/tensorflow
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
NVCC_INC=/usr/local/cuda/include/

OS=`uname -s`
if [[ ${OS} == "Darwin"  ]];then
    NVCC_LIB=/usr/local/cuda/lib/
elif [[ ${OS} == "Linux"  ]];then
    NVCC_LIB=/usr/local/cuda/lib64/
else
    echo "Other OS: ${OS}, Not supported"
    exit
fi

CC=g++

${CC} -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so  -shared -fPIC -I${TF_INC} -I${NVCC_INC} -L${TF_LIB} -L${NVCC_LIB} -ltensorflow_framework -lcudart -O2 -D_GLIBCXX_USE_CXX11_ABI=0

