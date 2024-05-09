To compile the shim and test program, just run `./compile.sh`
Note that you must have the CUDA Runtime Toolkit installed. Set the link/include path in compile.sh according to your CUDA install path.
To run the test program run `LD_PRELOAD=./shim.so ./cuda_lock_stats`

