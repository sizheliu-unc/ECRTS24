### 0. Requirements
This section requires the user to have sudo privilage to use the SCHED_FIFO policy. If the user do not have such permission, conducting this experiment within a container or VM (with sudo permission) will suffice.

This section does not require users to have a GPU. Instead, any machine with Linux should be able to perform this experiment.

### 1. Setup
To set up, run (we recommend running this outside of the docker container)
```
./setup.sh
```

This will create several folders (build, dist, glibc), among which only the dist folder will be used for our experiment.
Specifically, the set up will download the git repo for glibc-2.38, patched with our phase-fair-boosting.patch and build a custom glibc with it (located in dist).

### 2. Experiment
To run all the experiments, run inside the docker container:
```
./run_all.sh
```

This will generate several trace files inside the trace/ folder:
- mutex_test_1_thread_ni.html
    - One thread, using pthread_mutex without PRIO_INHERIT policy
- mutex_test_2_thread_ni.html
    - Two threads, using the same pthread_mutex without PRIO_INHERIT policy
- rwlock_test_1_thread.html
    - One thread, using the default pthread_rwlock
- rwlock_test_4_thread.html
    - Four threads, using the same default pthread_rwlock
- mutex_test_1_thread_pi.html
    - One thread, using pthread_mutex with PRIO_INHERIT policy
- mutex_test_2_thread_pi.html
    - Two threads, using the same pthread_mutex with PRIO_INHERIT policy
- rwlock_test_1_thread_patched.html
    - One thread, using the custom pthread_rwlock
- rwlock_test_4_thread_patched.html
    - Four threads, using the same custom pthread_rwlock

### 3. How to interpret the results

For tests that runs with only 1 thread, the main goal is to evaluate the lock/unlock time under no contention (fast path). 

For tests that runs with more than 1 thread, the main goal is to evaluate the lock/unlock time under contention.

Particularly for rwlock, there are different types of contention, which will involve different futex operations

Read lock: 
- futex #1: (*phase-fair*) when a writer is pending lock, and the current lockholder is a reader.
- futex #2: when the current lockholder is a writer.

Write lock:
- futex #1: when the writer associated with this write request is not (cannot become) a primary writer (i.e. the top writer).
- futex #2: when the current lockholder is a reader.

Read unlock:
- wakeup #1: wake up pending primary writer, if exists.
- wakeup #2: (*phase-fair*) wake up readers blocked due to read lock futex #1

Write unlock:
- wakeup #1: wake up pending readers.
- wakeup #2: wakeup one pending writer (to become the primary writer).

During trace analysis, it is recommended to find cases where all the mentioned futex are involved, as they are able to more accurately reflect the overhead involved.
