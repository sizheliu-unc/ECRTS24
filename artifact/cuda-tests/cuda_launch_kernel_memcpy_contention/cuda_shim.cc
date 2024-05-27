#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <unistd.h>
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "kutrace_lib.h"
#include <cuda.h>

//static cudaError_t(*real_cudaMemcpyAsync)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = NULL;
static cudaError_t(*real_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL; 
//static cudaError_t(*real_cudaStreamSynchronize)(cudaStream_t) = NULL; 
//static CUresult(*real_cuCtxCreate)(CUcontext* pctx, unsigned int flags, CUdevice dev) = NULL;

/*
static CUresult(*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = NULL;


extern "C" CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev)
{
	if (real_cuCtxCreate == NULL)
	{
		printf("cuCtxCreate not initialized\n");
		real_cuCtxCreate = 
			reinterpret_cast<CUresult(*)(CUcontext*, unsigned int, CUdevice) >
						(dlsym(RTLD_NEXT, "cuCtxCreate"));
		
		if (real_cuCtxCreate == NULL)
		{
			fprintf(stderr, "Error: Could not dlink to real cuCtxCreate\n");
		}
		printf("cuCtxCreate initialized\n");
	}
    printf("Inside cuCtxCreate shim\n");
	CUresult r = real_cuCtxCreate(pctx, flags, dev);
	return r;
}

extern "C" CUresult cuLaunchKernel( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )
{
	if (real_cuLaunchKernel == NULL)
	{
		printf("cuLaunchKernel not initialized\n");
		real_cuLaunchKernel = 
			reinterpret_cast<CUresult(*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) >
						(dlsym(RTLD_NEXT, "cuLaunchKernel"));
		
		if (real_cuLaunchKernel == NULL)
		{
			fprintf(stderr, "Error: Could not dlink to real cuLaunchKernel\n");
		}
		printf("cuLaunchKernel initialized\n");
	}
    printf("Inside cuLaunchKernel shim\n");
	kutrace::mark_b("driver");
	CUresult r = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
	kutrace::mark_b("/driver");
	return r;
}

extern "C" cudaError_t cudaStreamSynchronize ( cudaStream_t stream ) 
{
	if (real_cudaStreamSynchronize == NULL)
	{
		printf("cudaStreamSynchronize not initialized\n");
		real_cudaStreamSynchronize = 
			reinterpret_cast<cudaError_t(*)( cudaStream_t)>
						(dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
		
		if (real_cudaStreamSynchronize == NULL)
		{
			fprintf(stderr, "Error: Could not dlink to real cudaStreamSynchronize\n");
		}
		printf("cudaStreamSynchronize initialized\n");
	}
    printf("Inside cudaStreamSynchronize shim\n");
	kutrace::mark_b("cuss");
	cudaError_t r = real_cudaStreamSynchronize(stream);
	kutrace::mark_b("/cuss");
	return r;
}

extern "C" cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
{
	if (real_cudaMemcpyAsync == NULL)
	{
		printf("cudaMemcpyAsync not initialized\n");
		real_cudaMemcpyAsync = 
			reinterpret_cast<cudaError_t(*)( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t)>
						(dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
		
		if (real_cudaMemcpyAsync == NULL)
		{
			fprintf(stderr, "Error: Could not dlink to real cudaMemcpyAsync\n");
		}
		printf("cudaMemcpyAsync initialized\n");
	}

    printf("Inside cudaMemcpyAsync shim\n");
	kutrace::mark_b("cmcpa");
    cudaError_t r = real_cudaMemcpyAsync( dst, src, count, kind, str );
	kutrace::mark_b("/cmcpa");
	return r;
}
*/

extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
{
	if (real_cudaLaunchKernel == NULL)
	{
		printf("cudaLaunchKernel not initialized\n");
		
		real_cudaLaunchKernel = 
			reinterpret_cast<cudaError_t(*)(const void*, dim3, dim3, void**, size_t, cudaStream_t)>
						(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
		if (real_cudaLaunchKernel == NULL)
		{
			fprintf(stderr, "Error: Could not dlink to real cudaLaunchKernel\n");
		}
		printf("cudaLaunchKernel initialized\n");
	}

	kutrace::mark_a("culk");
	cudaError_t r = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
	kutrace::mark_a("/culk");
	return r;
}
