#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <tuple>
#include <chrono>
#include <ratio>
#include <ctime>
#include "cuda.h"
#include "cuda_runtime.h"

#define CUDA_CHECK(x) er = x; \
    if (er) {\
        std::cout << "Error!\n"; \
        std::cout << cudaGetErrorName(er) << "\n"; \
        std::cout << cudaGetErrorString(er) << std::endl; \
    }

__global__
void saxpy(int n, float a, float * x, float * y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__inline__ std::string getCudaRTVersion()
{
    std::string version_string;
    int version;
    int major, minor;
    cudaRuntimeGetVersion(&version);
    major = (int)((float)version/(float)1000);
    minor = (version - 1000*major)/10;
    version_string = std::to_string(major) + "." + std::to_string(minor);
    return version_string;
}

__inline__ void printCudaVersions()
{
    std::string version_string;
    int version;
    int major, minor;
    cudaRuntimeGetVersion(&version);
    major = (int)((float)version/(float)1000);
    minor = (version - 1000*major)/10;
    version_string = "CUDA runtime version: " + std::to_string(major) + "." + std::to_string(minor);
    std::cout << version_string << "\n";

    cudaDriverGetVersion(&version);
    major = (int)((float)version/(float)1000);
    minor = (version - 1000*major)/10;
    version_string = "CUDA driver version: " + std::to_string(major) + "." + std::to_string(minor);
    std::cout << version_string << std::endl;
}

int main(int argc, char** argv)
{
    printCudaVersions();

    unsigned int num_experiments = 5;

    unsigned int N = 1<<20;

    // Setup input data
    float* x = new float[N];
    float* y = new float[N];
    for (int i = 0; i < N; i++)
    {
        x[i] = i*2.5;
        y[i] = i*4.5;
    }

    float* d_x;
    float* d_y;

    // Setup memory on GPU and copy input data there
    cudaError_t er;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));


    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run twice just to page fault things in and cache stuff
    CUDA_CHECK(cudaMemcpyAsync(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Run twice just to page fault things in and cache stuff
    saxpy<<<4096,256,0,stream>>>(N, 2.0, d_x, d_y);
    cudaDeviceSynchronize();
    saxpy<<<4096,256>>>(N, 2.0, d_x, d_y);
    cudaDeviceSynchronize();

    std::vector<float> times;

    times.reserve(num_experiments);

    //uint64_t time;
    double timef;

    for (unsigned long i = 0; i < num_experiments; i++)
    {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        saxpy<<<4096,256>>>(N, 2.0, d_x, d_y);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> kernel_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);

        timef = kernel_runtime.count() * 1000000;
        
        times.emplace_back(timef);
    }

    std::cout << "Kernel Launch Times: \n";

    int run = 0;
    for (const auto& time : times)
    {
        std::cout << "Kernel launch #" << run++ << " took " << std::to_string(time) << "us\n";
    }

    std::cout << std::endl;

    cudaFree(d_x);
    delete[] x;

    return 0;
}
