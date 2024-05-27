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
#include "kutrace_lib.h"

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

	char buff[10];
	uintptr_t addr_p = (uintptr_t)mutex;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	kutrace::mark_a(buff);

	kutrace::mark_a("mt_lk");
	int r = real_pthread_mutex_lock(mutex);
	kutrace::mark_a("/mt_lk");

	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
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

	char buff[10];
	uintptr_t addr_p = (uintptr_t)mutex;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	kutrace::mark_a(buff);

	kutrace::mark_a("mt_ul");
	int r = real_pthread_mutex_unlock(mutex);
	kutrace::mark_a("/mt_ul");

	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
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

	char buff[10];
	uintptr_t addr_p = (uintptr_t)__rwlock;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	kutrace::mark_a(buff);
	kutrace::mark_b("rdlk");
	int r = real_pthread_rwlock_rdlock(__rwlock);
	kutrace::mark_b("/rdlk");
	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
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

	char buff[10];
	uintptr_t addr_p = (uintptr_t)rwlock;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	kutrace::mark_a(buff);
	kutrace::mark_b("trdlk");
	int r = real_pthread_rwlock_tryrdlock(rwlock);
	kutrace::mark_b("/trdlk");
	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
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

	char buff[10];
	uintptr_t addr_p = (uintptr_t)__rwlock;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	kutrace::mark_a(buff);
	kutrace::mark_b("wrlk");
	int r = real_pthread_rwlock_wrlock(__rwlock);
	kutrace::mark_b("/wrlk");

	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
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

	char buff[10];
	uintptr_t addr_p = (uintptr_t)__rwlock;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	kutrace::mark_a(buff);
	kutrace::mark_b("twrlk");
	int r = real_pthread_rwlock_trywrlock(__rwlock);
	kutrace::mark_b("/twrlk");
	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
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

	char buff[10];
	uintptr_t addr_p = (uintptr_t)__rwlock;
	unsigned int addr_i = ((unsigned int)addr_p) % 1000000;
	snprintf(buff, 10, "%u", addr_i);
	kutrace::mark_a(buff);
	kutrace::mark_b("rwulk");
	int r = real_pthread_rwlock_unlock(__rwlock);
	kutrace::mark_b("/rwulk");
	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
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
	kutrace::mark_a(buff);
	kutrace::mark_b("twl");
	//printf("Timed wrlock address is 0x%p, timeout.tv_sec=%ld, timeout.tv_nsec=%ld\n", rwlock, abs_timeout->tv_sec, abs_timeout->tv_nsec);
	int r = real_pthread_rwlock_timedwrlock(rwlock, abs_timeout);
	kutrace::mark_b("/twl");
	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
	return r;
	//return 22;
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
	kutrace::mark_a(buff);
	kutrace::mark_b("trl");
	//printf("Timed rdlock address is 0x%p\n", rwlock);
	int r = real_pthread_rwlock_timedrdlock(rwlock, abs_timeout);
	kutrace::mark_b("/trl");
	snprintf(buff, 10, "%d", r);
	kutrace::mark_a(buff);
	return r;
	//return 22;
}