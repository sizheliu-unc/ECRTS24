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

// 2100000 cycles is 1 msec on our 2.1GHz CPU
bool infer_start = false;
uint64_t num_call = 0;

#define CYCLES_PER_MS 2100000

uint64_t rdtsc_timestamp()
{
   uint32_t hi, lo;
   __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
   return ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
}

static int (*real_pthread_mutex_lock)		(pthread_mutex_t* mutex) = NULL;
static int (*real_pthread_mutex_unlock)		(pthread_mutex_t* mutex) = NULL;
static int (*real_pthread_rwlock_rdlock)	(pthread_rwlock_t *__rwlock) = NULL;
static int (*real_pthread_rwlock_tryrdlock) (pthread_rwlock_t *  rwlock) = NULL;
static int (*real_pthread_rwlock_wrlock) 	(pthread_rwlock_t *__rwlock) = NULL;
static int (*real_pthread_rwlock_trywrlock) (pthread_rwlock_t *__rwlock) = NULL;
static int (*real_pthread_rwlock_unlock)	(pthread_rwlock_t *__rwlock) = NULL;
static int (*real_pthread_rwlock_timedwrlock) (pthread_rwlock_t * rwlock, const struct timespec * abs_timeout) = NULL;
static int (*real_pthread_rwlock_timedrdlock) (pthread_rwlock_t * rwlock, const struct timespec * abs_timeout) = NULL;
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
		fprintf(stderr, "Initialized pthread_mutex_lock\n");
	}

	return real_pthread_mutex_lock(mutex);
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
	if (infer_start) {
		num_call++;
	}
	return real_pthread_mutex_unlock(mutex);
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

	return real_pthread_rwlock_rdlock(__rwlock);
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

	return real_pthread_rwlock_tryrdlock(rwlock);
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

	return real_pthread_rwlock_wrlock(__rwlock);
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

	return real_pthread_rwlock_trywrlock(__rwlock);
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

	int r = real_pthread_rwlock_unlock(__rwlock);
	return r;
}

extern "C" int pthread_rwlock_timedwrlock(pthread_rwlock_t * rwlock, const struct timespec * abs_timeout)
{
	if (real_pthread_rwlock_timedwrlock == NULL)
	{
		fprintf(stderr, "pthread_rwlock_timedwrlock not initialized\n");
		real_pthread_rwlock_timedwrlock = reinterpret_cast<int(*)(pthread_rwlock_t *, const struct timespec *)>(dlsym(RTLD_NEXT, "pthread_rwlock_timedwrlock"));
		if (real_pthread_rwlock_timedwrlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_rwlock_timedwrlock()\n");
			exit(255);
		}
		fprintf(stderr,"Initialized pthread_rwlock_timedwrlock\n");
	}

	char buff[10];
	uintptr_t addr_p = (uintptr_t)rwlock;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	int r = real_pthread_rwlock_timedwrlock(rwlock, abs_timeout);
	snprintf(buff, 10, "%d", r);
	return r;
}

extern "C" int pthread_rwlock_timedrdlock(pthread_rwlock_t * rwlock, const struct timespec * abs_timeout)
{
	if (real_pthread_rwlock_timedrdlock == NULL)
	{
		fprintf(stderr, "pthread_rwlock_timedrdlock not initialized\n");
		real_pthread_rwlock_timedrdlock = reinterpret_cast<int(*)(pthread_rwlock_t *, const struct timespec *)>(dlsym(RTLD_NEXT, "pthread_rwlock_timedrdlock"));
		if (real_pthread_rwlock_timedrdlock == NULL)
		{
			fprintf(stderr,"Couldn't find real pthread_rwlock_timedrdlock()\n");
			exit(255);
		}
		fprintf(stderr,"Initialized pthread_rwlock_timedrdlock\n");
	}

	char buff[10];
	uintptr_t addr_p = (uintptr_t)rwlock;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	int r = real_pthread_rwlock_timedrdlock(rwlock, abs_timeout);
	snprintf(buff, 10, "%d", r);
	return r;
}
