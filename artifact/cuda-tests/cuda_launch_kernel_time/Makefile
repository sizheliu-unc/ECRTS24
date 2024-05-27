NVCC = /usr/local/cuda-12.2.2/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda-12.2.2/include 
CUDA_LIB = /usr/local/cuda-12.2.2/targets/x86_64-linux/lib 
KUTRACE_DIR = ../kutrace_min

.PHONY: all clean

all: cuda_launch_kernel_time

cuda_launch_kernel_time: cuda_launch_kernel_time.cu $(KUTRACE_DIR)/kutrace_lib.cc
	$(NVCC) -I$(CUDA_INCLUDE) -L$(CUDA_LIB) -I$(KUTRACE_DIR) -lcudart $^ -o $@

run: cuda_launch_kernel_time
	./cuda_launch_kernel_time
	./process.sh cuda_launch_kernel_time.trace

clean:
	rm -rf *.trace* cuda_launch_kernel_time