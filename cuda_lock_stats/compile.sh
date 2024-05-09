g++ -I/usr/local/cuda-12.2.2/include -O0 -fPIC -shared -g -o shim.so shim.cc -ldl -L/usr/local/cuda-12.2.2/targets/x86_64-linux/lib -lcudart -I.
/usr/local/cuda-12.2.2/bin/nvcc -I/usr/local/cuda-12.2.2/include cuda_lock_stats.cu -g -o cuda_lock_stats -O0 -L/usr/local/cuda-12.2.2/targets/x86_64-linux/lib -lcudart -Isrc
