#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <unistd.h>
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstdlib>
#include <string>
#include <pthread.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <ratio>
#include <ctime>
#include <tabulate.hpp>
#include <string>
#include <fstream>
using namespace tabulate;

enum lock_type { UNSET, MUTEX, WLOCK, RLOCK, TWLOCK, TRLOCK};

struct data
{
	lock_type type;
    void* addr;
    //uint64_t last_start;
	std::chrono::high_resolution_clock::time_point last_start; 
    double count;
    double sum;
};

typedef struct data data_t;

#define CAPACITY 50000 // Size of the HashTable.
data_t ht[CAPACITY];
int started;

static int (*real_pthread_mutex_lock)		(pthread_mutex_t* mutex) = NULL;
static int (*real_pthread_mutex_unlock)		(pthread_mutex_t* mutex) = NULL;
static int (*real_pthread_rwlock_rdlock)	(pthread_rwlock_t *__rwlock) = NULL;
static int (*real_pthread_rwlock_tryrdlock) (pthread_rwlock_t *  rwlock) = NULL;
static int (*real_pthread_rwlock_wrlock) 	(pthread_rwlock_t *__rwlock) = NULL;
static int (*real_pthread_rwlock_trywrlock) (pthread_rwlock_t *__rwlock) = NULL;
static int (*real_pthread_rwlock_unlock)	(pthread_rwlock_t *__rwlock) = NULL;

static cudaError_t(*real_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL; 

static int hash_idx (void* lock)
{

    uint64_t num = (uint64_t)lock;
    int count = 0;
    while (num > 0)
    {
        count += (num % 10);
        num /= 10;
    } 

    return count % CAPACITY;
}

extern "C" int pthread_mutex_lock (pthread_mutex_t* mutex)
{
	if (real_pthread_mutex_lock == NULL)
	{
		fprintf(stderr, "pthread_mutex_lock not initialized\n");
		real_pthread_mutex_lock = reinterpret_cast<int(*)(pthread_mutex_t*)>(dlsym(RTLD_NEXT, "pthread_mutex_lock"));
		if (real_pthread_mutex_lock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_mutex_lock()\n");
			exit(255);
		}
        //ht = (uint64_t*)malloc(sizeof(uint64_t) * CAPACITY);
        for (int i = 0; i < CAPACITY; i++)
        {
			ht[i].type = lock_type::UNSET;
            ht[i].addr = NULL;
            ht[i].last_start = std::chrono::high_resolution_clock::now();
            ht[i].count = 0; 
            ht[i].sum = 0;
        }
        started = 0;
		fprintf(stderr, "Initialized pthread_mutex_lock\n");
	}


    if (started == 1)
    {    
		ht[hash_idx(mutex)].type = lock_type::MUTEX;
        ht[hash_idx(mutex)].addr = mutex;
        ht[hash_idx(mutex)].last_start = std::chrono::high_resolution_clock::now();
    }
	int r = real_pthread_mutex_lock(mutex);

	return r;
}

extern "C" int pthread_mutex_unlock (pthread_mutex_t* mutex)
{
	if (real_pthread_mutex_unlock == NULL)
	{
		fprintf(stderr, "pthread_mutex_unlock not initialized\n");
		real_pthread_mutex_unlock = reinterpret_cast<int(*)(pthread_mutex_t*)>(dlsym(RTLD_NEXT, "pthread_mutex_unlock"));
		if (real_pthread_mutex_unlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_mutex_unlock()\n");
			exit(255);
		}
		fprintf(stderr, "Initialized pthread_mutex_unlock\n");
	}

    if (started == 1)
    {
        int i = hash_idx(mutex);
        if (ht[hash_idx(mutex)].addr != NULL)
        {
			std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration_locked = std::chrono::duration_cast<std::chrono::duration<double>>(now-ht[hash_idx(mutex)].last_start);
			ht[hash_idx(mutex)].sum += duration_locked.count()*1000000;
            ht[hash_idx(mutex)].count += 1;
        }
    }

	int r = real_pthread_mutex_unlock(mutex);

	return r;
}


__attribute__((destructor))
static void shutdown() 
{

	Table stats;
	stats.add_row({"Type", "Address", "Count", "Avg Time Held (us)"});

    printf("=================================Locking Summary=================================\n");
    for (int i = 0; i < CAPACITY; i++)
    {
        if (ht[i].addr != NULL)
        {
			std::string type;
			if (ht[i].type == lock_type::MUTEX)
			{
				type = "Mutex";
			}
			else if (ht[i].type == lock_type::RLOCK)
			{
				type = "Rd Lock";
			}
			else if (ht[i].type == lock_type::WLOCK)
			{
				type = "Wr Lock";
			}
			else if (ht[i].type == lock_type::TRLOCK)
			{
				type = "Timed Rd Lock";
			}
			else if (ht[i].type == lock_type::TWLOCK)
			{
				type = "Timed Wr Lock";
			}
			else 
			{
				type = "Unknown";
			}
			std::stringstream hex_addr;
			hex_addr << std::hex << ht[i].addr;
			double avg_time_held = ht[i].sum / ht[i].count;
			stats.add_row({type, hex_addr.str(), std::to_string(int(ht[i].count)), std::to_string(avg_time_held)});
        }
    }
	  // center-align and color header cells
  	for (size_t i = 0; i < 4; ++i) {
    	stats[0][i].format()
      .font_color(Color::yellow)
      .font_align(FontAlign::center)
      .font_style({FontStyle::bold});
  	}

	std::cout << stats << std::endl;
}



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

    started = 1;
	cudaError_t r = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    started = 2;
	return r;
}
extern "C" int pthread_rwlock_rdlock(pthread_rwlock_t *__rwlock)
{
	if (real_pthread_rwlock_rdlock == NULL)
	{
		fprintf(stderr, "pthread_rwlock_rdlock not initialized\n");
		real_pthread_rwlock_rdlock = reinterpret_cast<int(*)(pthread_rwlock_t*)>(dlsym(RTLD_NEXT, "pthread_rwlock_rdlock"));
		if (real_pthread_rwlock_rdlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_rwlock_rdlock()\n");
			exit(255);
		}
		fprintf(stderr,"Initialized pthread_rwlock_rdlock\n");
	}

	if (started == 1)
    {    
		ht[hash_idx(__rwlock)].type = lock_type::RLOCK;
        ht[hash_idx(__rwlock)].addr = __rwlock;
        ht[hash_idx(__rwlock)].last_start = std::chrono::high_resolution_clock::now();
    }

	int r = real_pthread_rwlock_rdlock(__rwlock);
	return r;
}

extern "C" int pthread_rwlock_tryrdlock(pthread_rwlock_t *rwlock)
{
	if (real_pthread_rwlock_tryrdlock == NULL)
	{
		fprintf(stderr, "pthread_rwlock_tryrdlock not initialized\n");
		real_pthread_rwlock_tryrdlock = reinterpret_cast<int(*)(pthread_rwlock_t*)>(dlsym(RTLD_NEXT, "pthread_rwlock_tryrdlock"));
		if (real_pthread_rwlock_tryrdlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_rwlock_tryrdlock()\n");
			exit(255);
		}
		fprintf(stderr,"Initialized pthread_rwlock_tryrdlock\n");
	}

	if (started == 1)
    {    
		ht[hash_idx(rwlock)].type = lock_type::TRLOCK;
        ht[hash_idx(rwlock)].addr = rwlock;
        ht[hash_idx(rwlock)].last_start = std::chrono::high_resolution_clock::now();
    }

	int r = real_pthread_rwlock_tryrdlock(rwlock);
	return r;
}

extern "C" int pthread_rwlock_wrlock(pthread_rwlock_t *__rwlock)
{
	if (real_pthread_rwlock_wrlock == NULL)
	{
		fprintf(stderr, "pthread_rwlock_wrlock not initialized\n");
		real_pthread_rwlock_wrlock = reinterpret_cast<int(*)(pthread_rwlock_t*)>(dlsym(RTLD_NEXT, "pthread_rwlock_wrlock"));
		if (real_pthread_rwlock_wrlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_rwlock_wrlock()\n");
			exit(255);
		}
		fprintf(stderr,"Initialized pthread_rwlock_wrlock\n");
	}

	if (started == 1)
    {    
		ht[hash_idx(__rwlock)].type = lock_type::WLOCK;
        ht[hash_idx(__rwlock)].addr = __rwlock;
        ht[hash_idx(__rwlock)].last_start = std::chrono::high_resolution_clock::now();
    }

	int r = real_pthread_rwlock_wrlock(__rwlock);
	return r;
}

extern "C" int pthread_rwlock_trywrlock(pthread_rwlock_t *__rwlock)
{
	if (real_pthread_rwlock_trywrlock == NULL)
	{
		fprintf(stderr, "pthread_rwlock_trywrlock not initialized\n");
		real_pthread_rwlock_trywrlock = reinterpret_cast<int(*)(pthread_rwlock_t*)>(dlsym(RTLD_NEXT, "pthread_rwlock_trywrlock"));
		if (real_pthread_rwlock_trywrlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_rwlock_trywrlock()\n");
			exit(255);
		}
		fprintf(stderr,"Initialized pthread_rwlock_trywrlock\n");
	}

	if (started == 1)
    {    
		ht[hash_idx(__rwlock)].type = lock_type::TWLOCK;
        ht[hash_idx(__rwlock)].addr = __rwlock;
        ht[hash_idx(__rwlock)].last_start = std::chrono::high_resolution_clock::now();
    }

	int r = real_pthread_rwlock_trywrlock(__rwlock);
	return r;
}

extern "C" int pthread_rwlock_unlock(pthread_rwlock_t *__rwlock)
{
	if (real_pthread_rwlock_unlock == NULL)
	{
		fprintf(stderr, "pthread_rwlock_unlock not initialized\n");
		real_pthread_rwlock_unlock = reinterpret_cast<int(*)(pthread_rwlock_t*)>(dlsym(RTLD_NEXT, "pthread_rwlock_unlock"));
		if (real_pthread_rwlock_unlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_rwlock_unlock()\n");
			exit(255);
		}
		fprintf(stderr,"Initialized pthread_rwlock_unlock\n");
	}

	 if (started == 1)
    {
        int i = hash_idx(__rwlock);
        if (ht[i].addr != NULL)
        {
			std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration_locked = std::chrono::duration_cast<std::chrono::duration<double>>(now-ht[i].last_start);
			ht[i].sum += duration_locked.count()*1000000;
            ht[i].count += 1;
		}
	}

	int r = real_pthread_rwlock_unlock(__rwlock);
	return r;
}

