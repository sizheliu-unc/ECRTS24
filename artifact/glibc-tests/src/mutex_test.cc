#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <cassert>
#include "kutrace_lib.h"
#include <thread>
#include <chrono>

// Define a global real-time mutex
pthread_mutex_t myMutex;// = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t myBarrier;

// Function to be executed by each thread
void* threadFunction(void* threadID) {
    long tid = (long)threadID;

    // Wait for all threads to reach the barrier
    pthread_barrier_wait(&myBarrier);

    for (int i = 0; i < 10; i++)
    {
        // Acquire the mutex
        kutrace::mark_a("lock");
        if (pthread_mutex_lock(&myMutex) != 0) {
            std::cerr << "Error locking mutex in thread " << tid << std::endl;
            pthread_exit(NULL);
        }
        kutrace::mark_a("/lock");

        // Critical section - Simulating some work
        std::cout << "Thread " << tid << " has acquired the mutex" << std::endl;
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));

        // Release the mutex
        kutrace::mark_a("unlock");
        if (pthread_mutex_unlock(&myMutex) != 0) {
            std::cerr << "Error unlocking mutex in thread " << tid << std::endl;
            pthread_exit(NULL);
        }
        kutrace::mark_a("/unlock");

        std::cout << "Thread " << tid << " has released the mutex" << std::endl;
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    bool use_pi = false;
    if (argc < 2) {
        std::cerr << "No valid argument given. Use pi/ni (prio-inherit/no-inherit) as the argument!" << std::endl;
        return 0;        
    }
    std::string arg(argv[1]);
    if (arg == "pi") {
        use_pi = true;
    }
    int NUM_THREADS = 2;
    if (argc == 3) {
        NUM_THREADS = atoi(argv[2]);
        assert(NUM_THREADS > 0);
    }

    pthread_mutexattr_t attrs;
    if (pthread_mutexattr_init(&attrs))
    {
        std::cout << "Failed to init mutex attributes" << std::endl;
        pthread_exit(NULL);
    }
    if (pthread_mutexattr_setprotocol(&attrs, PTHREAD_PRIO_INHERIT))
    {
        std::cout << "Failed to configure mutex attr protocol" << std::endl;
        pthread_exit(NULL);
    }
    if ((use_pi && pthread_mutex_init(&myMutex, &attrs)) || (!use_pi && pthread_mutex_init(&myMutex, NULL))) 
    {
        std::cout << "Failed to init mutex" << std::endl;
        pthread_exit(NULL);
    }



    pthread_t threads[NUM_THREADS];
    int rc;
    
    // Initialize the barrier
    pthread_barrier_init(&myBarrier, NULL, NUM_THREADS + 1);

    // Create two threads
    for (long i = 0; i < NUM_THREADS; ++i) {
        pthread_attr_t pthread_attr;
        pthread_attr_init(&pthread_attr);
        pthread_attr_setinheritsched(&pthread_attr, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy(&pthread_attr, SCHED_FIFO);
        struct sched_param prio;
        prio.sched_priority = 30 + i;
        pthread_attr_setschedparam(&pthread_attr, &prio);
        rc = pthread_create(&threads[i], &pthread_attr, threadFunction, (void*)i);
        if (rc) {
            std::cerr << "Error creating thread " << i << ". Return code: " << rc << std::endl;
            return 1;
        }
    }

    kutrace::go("mutex_wait_time");
    kutrace::mark_a("test");
    // Wait for all threads to reach the barrier before releasing them
    pthread_barrier_wait(&myBarrier);

    // Join threads to wait for their completion
    for (long i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    kutrace::mark_a("/test");
    kutrace::stop("mutex_test_result.trace");
    pthread_barrier_destroy(&myBarrier);
    // Destroy the mutex
    pthread_mutex_destroy(&myMutex);
    if (pthread_mutexattr_destroy(&attrs) != 0) {
        std::cout << "pthread_mutex_attr_destroy() error" << std::endl;
        pthread_exit(NULL);
    }



    return 0;
}

