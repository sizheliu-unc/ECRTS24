CC = /usr/bin/g++
SHIM_CC_FLAGS = -std=c++11 -fPIC -O0 -shared -g -ldl
NVCC = /usr/local/cuda-12.2.2/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda-12.2.2/include 
CUDA_LIB = /usr/local/cuda-12.2.2/targets/x86_64-linux/lib 
KUTRACE_DIR = ../../kutrace_min

.PHONY: all clean

all: cuda_launch_kernel_memcpy_contention shim

cuda_launch_kernel_memcpy_contention: cuda_launch_kernel_memcpy_contention.cu $(KUTRACE_DIR)/kutrace_lib.cc
	$(NVCC) -I$(CUDA_INCLUDE) -L$(CUDA_LIB) -I$(KUTRACE_DIR) -lcudart $^ -o $@

shim: pthread_shim.cc cuda_shim.cc
	g++ $(SHIM_CC_FLAGS) -I$(CUDA_INCLUDE) -o shim.so pthread_shim.cc cuda_shim.cc -ldl -L$(CUDA_LIB) -lcudart -I$(KUTRACE_DIR) $(KUTRACE_DIR)/kutrace_lib.cc

run: cuda_launch_kernel_memcpy_contention shim
	LD_PRELOAD=./shim.so ./cuda_launch_kernel_memcpy_contention
	./process.sh cuda_launch_kernel_memcpy_contention.trace

clean:
	rm -rf *.trace* cuda_launch_kernel_memcpy_contention shim.so
