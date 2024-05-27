#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ratio>
#include <ctime>
#include <vector>
#include <algorithm>

#include "kutrace_lib.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define CUDA_CHECK(x) er = x; \
    if (er) {\
        std::cout << "Error!\n"; \
        std::cout << cudaGetErrorName(er) << "\n"; \
        std::cout << cudaGetErrorString(er) << std::endl; \
    }

void stats(std::vector<double>& v)
{
    if (v.empty()) return;

    std::sort(v.begin(), v.end());

    std::cout << "Num elements = " << v.size() << "\n";

    double sum = 0;
    for (const double d : v)
    {
        sum += d;
    } 
    double mean = sum / v.size();
    std::cout << "Mean = " << mean << " us\n";

    double median;
    if (v.size() % 2 == 0)
    {
        median = (v.at(v.size()/2 - 1) + v.at(v.size()/2)) / 2;
    }
    else 
    {
        median = v.at(v.size()/2);
    }
    std::cout << "Median = " << median << " us\n";

    std::cout << "Min = " << v.at(0) << " us\n";

    int q1_idx = v.size() / 4;
    double q1 = v.at(q1_idx);
    std::cout << "Q1 = " << q1 << " us\n";

    int q3_idx = v.size() * 3 / 4;
    double q3 = v.at(q3_idx);
    std::cout << "Q3 = " << q3 << " us\n";

    std::cout << "Max = " << v.at(v.size()-1) << " us\n";

}

__global__
void saxpy(int n, float * x, float * y, float * z)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    const float a = 2.0;  
    if (i < n) z[i] = a*x[i] + y[i];
}

using hr_delta = std::chrono::duration<double>;
using hr_time = std::chrono::high_resolution_clock::time_point;

int main(int argc, char** argv)
{

    // Durations are in microseconds
    std::vector<double> h2d_times;
    std::vector<double> d2h_times;

    hr_time start, end;

    const unsigned int num_data_elts = 1<<20;

    float *d_x, *d_y, *d_z;
    float* x = new float[num_data_elts];
    float* y = new float[num_data_elts];
    float* z = new float[num_data_elts];
    
    cudaError_t er;
    CUDA_CHECK(cudaMalloc(&d_x, num_data_elts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, num_data_elts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, num_data_elts * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < num_data_elts; i++)
    {
        x[i] = i * 2.5;
        y[i] = i * 4.5;
    }

    // Do the first run before we start the test, so we page everything in
    CUDA_CHECK(cudaMemcpyAsync(d_x, x, num_data_elts, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y, y, num_data_elts, cudaMemcpyHostToDevice, stream));
    cudaDeviceSynchronize();
    saxpy<<<4096, 256, 0, stream>>>(num_data_elts, d_x, d_y, d_z);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(z, d_z, num_data_elts, cudaMemcpyDeviceToHost));


    kutrace::go("cuda_memcpy_time");

    // Take the median of 100 runs.
    // Do an extra run in the beginning to make sure everything is paged in
    kutrace::mark_c("test");
    for (int i = 0; i < 100; i++)
    {
        // Copy data into x vector
        kutrace::mark_b("h2d_x");
        start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(d_x, x, num_data_elts, cudaMemcpyHostToDevice, stream));
        end = std::chrono::high_resolution_clock::now();
        kutrace::mark_b("/h2d_x");
        hr_delta duration = std::chrono::duration_cast<hr_delta>(end - start);
        h2d_times.push_back(duration.count()*1000000);

        // Copy data into y vector
        kutrace::mark_b("h2d_y");
        start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(d_y, y, num_data_elts, cudaMemcpyHostToDevice, stream));
        end = std::chrono::high_resolution_clock::now();
        kutrace::mark_b("/h2d_y");
        duration = std::chrono::duration_cast<hr_delta>(end - start);
        h2d_times.push_back(duration.count()*1000000);

        // Wait for all the data to copy over
        cudaDeviceSynchronize();

        // Run the kernel computation
        saxpy<<<4096, 256, 0, stream>>>(num_data_elts, d_x, d_y, d_z);

        // Wait for the kernel to finish the computation, before we try to copy the output to the CPU
        cudaDeviceSynchronize();

        // Copy the output back to the CPU
        kutrace::mark_b("d2h_z");
        start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(z, d_z, num_data_elts, cudaMemcpyDeviceToHost));
        end = std::chrono::high_resolution_clock::now();
        kutrace::mark_b("/d2h_z");
        duration = std::chrono::duration_cast<hr_delta>(end - start);
        d2h_times.push_back(duration.count()*1000000);
    }

    // End the test
    kutrace::mark_c("/test");

    // Free all resources
    CUDA_CHECK(cudaStreamDestroy(stream));

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    
    delete[] x;
    delete[] y;
    delete[] z;

    std::cout << "Host-to-Device Stats" << std::endl;
    std::cout << "--------------------" << std::endl;
    stats(h2d_times);
    std::cout << "\n\n";
    std::cout << "Device-to-Host Stats" << std::endl;
    std::cout << "--------------------" << std::endl;
    stats(d2h_times);
    std::cout << std::endl;

    kutrace::stop("cuda_memcpy_time.trace");
}
