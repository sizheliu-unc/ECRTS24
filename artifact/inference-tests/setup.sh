#!/bin/bash
alias trtexec=/usr/local/TensorRT-8.6.1.6/bin/trtexec
alias nsys=/usr/local/nsight/bin/nsys
alias nvcc=/usr/local/cuda-12.2.2/bin/nvcc

if test -d build; then
    echo "build directory exists, skip creating..."
else
    mkdir build
fi

if test -d lib; then
    echo "lib directory exists, skip creating..."
else
    mkdir lib
fi


export CUDACXX=/usr/local/cuda-12.2.2/bin/nvcc
cd src/
./shim_compile.sh
./dummy_lib_compile.sh

cd ../build/
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2.2 ..
make
rm -f shim.so model_launch_time.py
ln -s ../lib/shim.so shim.so
ln -s ../src/model_launch_time.py model_launch_time.py
cd ..
