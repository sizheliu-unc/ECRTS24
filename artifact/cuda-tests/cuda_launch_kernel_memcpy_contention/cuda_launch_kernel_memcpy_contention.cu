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

#define NUM_THREADS 50

pthread_barrier_t myBarrier;
bool initialized;

using hr_delta = std::chrono::duration<double>;
using hr_time = std::chrono::high_resolution_clock::time_point;

__global__
void dummy_kernel()
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    const float a = 2.0;  
    int j = i * a;
}

void* thread_func(void* thread_id)
{
    const int tid = *(int*) thread_id;
    const unsigned int num_data_elts = 1 << 20;

    if (initialized)
        pthread_barrier_wait(&myBarrier);

    float* x = new float[num_data_elts];
    float* y = new float[num_data_elts];
    
    cudaError_t er;
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, num_data_elts * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < num_data_elts; i++)
    {
        x[i] = i * 2.5;
    }

    // Take the median of 100 runs.
    // Do an extra run in the beginning to make sure everything is paged in
    char tid_buf[512];
    sprintf(tid_buf, "t%d", tid);
    kutrace::mark_c(tid_buf);
    for (int i = 0; i < 20; i++)
    {
        if (tid % 2)
        {
            kutrace::mark_b("h2d");
            // Copy data into x vector
            CUDA_CHECK(cudaMemcpyAsync(d_x, x, num_data_elts, cudaMemcpyHostToDevice, stream));
            kutrace::mark_b("/h2d");

            // Wait for all the data to copy over
            cudaDeviceSynchronize();

            kutrace::mark_b("d2h");
            // Copy the output back to the CPU
            CUDA_CHECK(cudaMemcpy(y, d_x, num_data_elts, cudaMemcpyDeviceToHost));
            kutrace::mark_b("/d2h");
        }
        else
        {
            // Run the kernel computation
            kutrace::mark_b("launch");
            dummy_kernel<<<4096, 256, 0, stream>>>();
            kutrace::mark_b("/launch");
        }

        // Wait for the kernel to finish the computation, before we try to copy the output to the CPU
        cudaDeviceSynchronize();

    }

    // End the test
    sprintf(tid_buf, "/t%d", tid);
    kutrace::mark_c(tid_buf);

    // Free all resources
    CUDA_CHECK(cudaStreamDestroy(stream));

    cudaFree(d_x);
    
    delete[] x;
    delete[] y;
}

int main(int argc, char** argv)
{
    pthread_barrier_init(&myBarrier, NULL, NUM_THREADS + 1);

    // Temp thread/kernel launch/memcpy to page everything in
    initialized = 0;
    pthread_t warmup_launch_thread, warmup_memcpy_thread;
    int warmup_launch_id = 0;
    int warmup_memcpy_id = 1;

    pthread_create(&warmup_launch_thread, NULL, thread_func, &warmup_launch_id);
    pthread_create(&warmup_memcpy_thread, NULL, thread_func, &warmup_memcpy_id);

    pthread_join(warmup_launch_thread, NULL);
    pthread_join(warmup_memcpy_thread, NULL);
    initialized = 1;

    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    kutrace::go("cuda_launch_kernel_memcpy_contention");
    // Start the test
    kutrace::mark_c("test");
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
    }

    pthread_barrier_wait(&myBarrier);

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&myBarrier);
    // End the test
    kutrace::mark_c("/test");
    kutrace::stop("cuda_launch_kernel_memcpy_contention.trace");
}
