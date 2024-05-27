#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <cassert>
#include "kutrace_lib.h"
#include <thread>
#include <chrono>

// Define a global real-time mutex
pthread_barrier_t myBarrier;
pthread_rwlock_t lock;

// Function to be executed by each thread
void* readerThreadFunction(void* threadID) {
    long tid = (long)threadID;

    std::cout << "Reader thread " << tid << " started" << std::endl;

    // Wait for all threads to reach the barrier
    pthread_barrier_wait(&myBarrier);

    for (int i = 0; i < 10; i++)
    {
        // Acquire the mutex
        kutrace::mark_a("rd");
        auto err = pthread_rwlock_rdlock(&lock);
        kutrace::mark_a("/rd");
        if (err != 0) {
            std::cerr << "Error rdlocking lock in thread " << tid << std::endl;
            pthread_exit(NULL);
        }

        // Critical section - Simulating some work
        std::cout << "Thread " << tid << " has acquired the lock in rd mode" << std::endl;
        std::this_thread::sleep_for(std::chrono::nanoseconds(5000));

        // Release the mutex
        kutrace::mark_a("unlock");
        err = pthread_rwlock_unlock(&lock);
        kutrace::mark_a("/unlock");
        if (err != 0) {
            std::cerr << "Error unlocking rwlock in thread " << tid << std::endl;
            pthread_exit(NULL);
        }

        std::cout << "Thread " << tid << " has released the rwlock" << std::endl;
    }

    pthread_exit(NULL);
}

void* writerThreadFunction(void* threadID) {
    long tid = (long)threadID;

    std::cout << "Writer thread " << tid << " started" << std::endl;

    // Wait for all threads to reach the barrier
    pthread_barrier_wait(&myBarrier);

    for (int i = 0; i < 10; i++)
    {
        kutrace::mark_a("wr");
        auto err = pthread_rwlock_wrlock(&lock);
        kutrace::mark_a("/wr");
        // Acquire the mutex
        if (err != 0) {
            std::cerr << "Error wrlocking lock in thread " << tid << std::endl;
            pthread_exit(NULL);
        }

        // Critical section - Simulating some work
        std::cout << "Thread " << tid << " has acquired the lock in wr mode" << std::endl;
        std::this_thread::sleep_for(std::chrono::nanoseconds(5000));

        kutrace::mark_a("unlock");
        err = pthread_rwlock_unlock(&lock);
        kutrace::mark_a("/unlock");
        // Release the mutex
        if (err != 0) {
            std::cerr << "Error unlocking rwlock in thread " << tid << std::endl;
            pthread_exit(NULL);
        }

        std::cout << "Thread " << tid << " has released the rwlock" << std::endl;
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    int NUM_THREADS = 4;
    if (argc >= 1) {
        NUM_THREADS = atoi(argv[1]);
        assert(NUM_THREADS > 0);
    }
    if (pthread_rwlock_init(&lock, NULL)) 
    {
        std::cout << "Failed to init rwlock" << std::endl;
        pthread_exit(NULL);
    }


    pthread_attr_t attr;
    pthread_t threads[NUM_THREADS];
    int rc;
    
    // Initialize the barrier
    pthread_barrier_init(&myBarrier, NULL, NUM_THREADS + 1);

    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    struct sched_param prio;
    prio.sched_priority = 30;
    pthread_attr_setschedparam(&attr, &prio);

    // Create two threads
    for (long i = 0; i < NUM_THREADS; ++i) {
        if (i%2 == 0)
            rc = pthread_create(&threads[i], &attr, writerThreadFunction, (void*)i);
        else
            rc = pthread_create(&threads[i], &attr, readerThreadFunction, (void*)i);
        if (rc) {
            std::cerr << "Error creating thread " << i << ". Return code: " << rc << std::endl;
            return 1;
        }
    }

    kutrace::go("rwlock_wait_time");
    kutrace::mark_a("test");
    // Wait for all threads to reach the barrier before releasing them
    pthread_barrier_wait(&myBarrier);

    // Join threads to wait for their completion
    for (long i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    kutrace::mark_a("/test");
    kutrace::stop("rwlock_test_result.trace");

    pthread_barrier_destroy(&myBarrier);
    pthread_rwlock_destroy(&lock);

    return 0;
}

