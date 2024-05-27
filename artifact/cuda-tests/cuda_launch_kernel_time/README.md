# CUDA Launch Kernel Time
This test times the CUDA kernel launch times over 100 iterations of Saxpy, and prints out statistics.

**Note**: By default this runs on CUDA Runtime version 12.2.2 on the NVIDIA Driver 550.54.14.\
Changing the driver version requires sudo access, so to change it email wagle@cs.unc.edu.

# Building code
To build the test, run ```make```.

# Running code
To run the test and post-process the output for viewing, run ```make run```
This will print out the statistics of the CUDA kernel launch times 
