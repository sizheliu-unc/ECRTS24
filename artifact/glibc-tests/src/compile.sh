#!/bin/bash
GLIBC_DIR=$(realpath ../dist)
g++ rwlock_test.cc -std=c++11 -lpthread -o ../bin/rwlock_test -I../../kutrace_min ../../kutrace_min/kutrace_lib.cc
cp ../bin/rwlock_test ../bin/rwlock_test_patched
#patchelf --set-interpreter ${GLIBC_DIR}/lib64/ld-linux-x86-64.so.2 --set-rpath ${GLIBC_DIR}/lib64/ ../bin/rwlock_test_patched
g++ mutex_test.cc -std=c++11 -lpthread -o ../bin/mutex_test -I../../kutrace_min ../../kutrace_min/kutrace_lib.cc
