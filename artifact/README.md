Artifact Evaluation: "Autonomy Today: Many Delay-Prone Black Boxes"
=======
<!-- **Authors: Sizhe Liu**&dagger;, **Rohan Wagle**&dagger;, **James H. Anderson**&dagger;, **Ming Yang**&Dagger;, **Chi Zhang**&Dagger;, and **Yunhua Li**&Dagger;

&dagger;University of North Carolina at Chapel Hill; &Dagger;Weride Corp. -->

This document serves as instructions to perform artifact evaluation for paper titled "Autonomy Today: Many Delay-Prone Black Boxes" submitted to ECRTS24.

# 0. Hardware
For artifact evaluation, we provide access to our hardware via ssh. 

# 1. CUDA tests
CUDA tests are located inside the `artifact/cuda-tests` folder.
## 1.1 CUDA Kernel Launch-Memcpy Contention Test
This test demonstrates the contention that can occur between pthread rwlocks and mutexes when many CUDA memcopies and CUDA kernels are launched in parallel.
This test spawns 50 pthreads. Of the 50 threads, 25 will perform 20 CUDA kernel launches, and the other 25 threads perform 20 Host-to-Device memcopies and 20 Device-to-Host memcopies.

### Building code
To build the test, run ```make```.

### Running code
To run the test and post-process the output for viewing, run ```make run```\
This will produce an html file called ```cuda_launch_kernel_memcpy_contention.trace.trimmed.html```\
This html file can be copied back to your host machine via WinSCP, FileZilla, Cyberduck, etc, and opened in any web browser.

### Viewing the trace output
The html file will be a bit big, so it might take a minute to load in the browser.

We use the KUtrace tracing framework written by Dr. Richard L. Sites.\
The full source code for KUtrace can be found here: https://github.com/dicksites/KUtrace

#### Reading KUtrace traces
Once it loads, you can click the `CPU` or `PID` dropdown buttons on the left panel, to open/collapse the CPU-view and Process-view of the trace, respectively.\
For reading this trace, we recommend viewing the Process-view instead of CPU-view.

You can increase the vertical spacing between process timelines in the trace by keeping your mouse over the list of processes on the left, and scrolling in.\
You can hover your mouse over the traces bars and scroll to zoom in and out of the trace itself (i.e. to scale time)

The stripped bars are the processes executing. Thick bars means the process is executing in kernel space, and thin bars means the process is in user space.\
If the process is not executing, its trace line will be either black, red, or yellow.\
The black bar means the process is idle, yellow means it's waiting for IO, and red means it's blocked waiting for a lock.

You can hold down the `Shift` + `left-click` on any of the trace bars to see what exactly that trace event means.
This is useful when you see a thick bar, because `Shift` + `left-click`ing it will tell you which Linux syscall the process entered the kernel through.

You can hold `Shift` + `left-click`, and drag the mouse to measure the time between events.\
If you let go of `Shift` before you let go of the mouse, then the time will remain on the screen.\
The red and blue flags with text or numbers in them are trace event markers our test code drops.\
For reference, the actual functions in our code that drop the markers are: kutrace::mark_a, kutrace::mark_b, kutrace::mark_c, where a,b,c just tell it what color to make the marker flag.\


### KUtrace marker meanings
In our trace, the markers have encoded meaning. The meanings are:

#### Pthread locks
- `wrlk` = Begin write-lock acquire

- `/wrlk` = Acquired write-lock

- `rdlk` = Begin read-lock acquire

- `/rdlk` = Acquired read-lock

- `mt_lk` = Begin mutex acquire

- `/mt_lk` = Acquired mutex

- `trdlk` = Begin try-read-lock acquire

- `/trdlk` = Acquired rdlock

- `twrlk` = Begin try-write-lock acquire

- `/twrlk` = Acquired write-lock

- `twl` = Begin timed-write-lock acquire

- `/twl` = Acquired write-lock

- `trl` = Begin timed-read-lock acquire

- `/trl` = Acquired read-lock

- `rwulk` = Begin unlocking read-write lock

- `/rwulk` = Unlocked read-write lock

Each of the markers denoting the beginning of a lock or unlock operation will be preceded by another marker in red containing a 6 digit number; this number is the address of the lock that's begin acquired (in decimal).
Each of the markers denoting the completion of a lock or unlock operation (the markers that start with a /) will be followed by another marker containing a number; this number is the return code of the lock/unlock function.

#### CUDA launch kernel

- `launch` = Begin *launching* CUDA kernel
- `/launch` = CPU done *launching* CUDA kernel - kernel will now begin running on GPU

**Note**: To be clear, this just tells us when the CUDA kernel gets launched, not necessarily when it actually gets executed on the GPU.
However, all we care about is the CPU-side contention between launching kernels and launching memcopying, so this marker is sufficient.

#### CUDA memcopy
- `h2d` = Begin async Host-to-Device memory copy launch
- `/h2d` = Async Host-to-Device memory copy launched

**Note**: To be clear, this just tells us when the asynchronous CPU->GPU memory copy is initiated, not when the data actually gets copied to the GPU.
However, all we care about is the CPU-side contention between launching kernels and launching memcopying, so this marker is sufficient.

- `d2h` = Begin async Device-to-Host memory copy launch
- `/d2h` = Async Device-to-Host memory copy launched

**Note**: To be clear, this just tells us when the asynchronous GPU->CPU memory copy is initiated, not when the data actually gets copied to the CPU.
However, all we care about is the CPU-side contention between launching kernels and launching memcopying, so this marker is sufficient.

## What you should see in the trace
This test shows the contention between CUDA memcopies (CPU <--> GPU) and the CUDA kernel launches.\
In particular, they contend for read-write locks and mutexes.\
You can see when a memcopy starts, with the 'h2d' and 'd2h' markers, and when they end, with '/h2d' and '/d2h', respectively.\
You can see when a CUDA kernel launch begins, with the 'launch' marker, and when it ends with the '/launch' marker.\
Between the (h2d,/h2d), (d2h,/d2h), and (launch,/launch) pairings, you will see all the pthread lock markers, showing what pthread lock functions the CUDA kernel launch and memcopy functions are using.\
You can see how threads will pause in the middle of a launch or memcopy operation, because they're blocked waiting for a lock.

## 1.2 CUDA Launch Kernel Time
This test times the CUDA kernel launch times over 100 iterations of Saxpy, and prints out statistics.

**Note**: By default this runs on CUDA Runtime version 12.2.2 on the NVIDIA Driver 550.54.14.\
Changing the driver version requires sudo access, so to change it email wagle@cs.unc.edu.

### Building code
To build the test, run ```make```.

### Running code
To run the test and post-process the output for viewing, run ```make run```
This will print out the statistics of the CUDA kernel launch times 

## 1.3 CUDA Memcpy Time
This test times the Host-to-Device memcopies and Device-to-Host memcopies over 100 iterations of Saxpy, and prints out statistics.

**Note**: By default this runs on CUDA Runtime version 12.2.2 on the NVIDIA Driver 550.54.14.\
Changing the driver version requires sudo access, so to change it email wagle@cs.unc.edu.

### Building code
To build the test, run ```make```.

### Running code
To run the test and post-process the output for viewing, run ```make run```
This will print out the statistics of the CUDA kernel launch times 

# 2. glibc tests
glibc tests are located inside the `artifact/glibc-tests` folder.
## 2.0 Requirements
This section requires the user to have sudo privilage to use the SCHED_FIFO policy. If the user do not have such permission, conducting this experiment within a container or VM (with sudo permission) will suffice.

We've setup a Docker container for this purpose. Enter the docker container with `docker exec -it ecrts24`. 
Then, enter this test directory with `cd artifact/glibc-tests/`

This section does not require users to have a GPU. Instead, any machine with Linux should be able to perform this experiment.

## 2.1 Setup
To set up, run (we recommend running this outside of the docker container)
```
./setup.sh
```

This will create several folders (build, dist, glibc), among which only the dist folder will be used for our experiment.
Specifically, the set up will download the git repo for glibc-2.38, patched with our phase-fair-boosting.patch and build a custom glibc with it (located in dist).

## 2.2 Experiment
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

## 2.3 How to interpret the results

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


# 3. DL inference tests
DL inference tests are located inside the `artifact/inference-tests` folder. This folder contains relevant material to evaluate deep learning inference, as well as CUDA graph.

## 3.0 Requirements
### Hardware requirements
For this section you will need the following hardware:
1 x NVIDIA GPU (Pascal generation or later)
x86-64 hardware platform (ARM should work as well, but is not guaranteed)

### Software requirements
Linux 5.4.0 or above
glibc 2.35 or above
CUDA Runtime 12.2.2
NVIDIA Driver 550.54.14
cuDNN 8.9.5.29
TensorRT 8.6.1.6
python 3.8

Both hardware and software will be supplied for the artifact evaluation.

### Extra dependency
For users wishing to test on their own hardware/software, extra steps will be necessary.

First, the user must change the paths to CUDA and TensorRT in CMakeLists.txt and setup.sh according to the download location of these components.

The user will also need to regenerate DL model engine plans (see models/README.md).

## 3.1 Setup
To compile the relevant inference experiments, run:

``` bash
./setup.sh
```

Make sure you are in the cuda-infer/ folder. This will generate several executable files and .so files.

build/infer
build/infer_once
build/shim.so

## 3.2 End-to-end execution time

To evaluate the end-to-end execution time of DL models, execute:
```bash
cd build/
./infer {model_name} {graph}
```

In which "{model_name}" must be one of the following:
vit, regnet, deit, detr, segformer; and "{graph}" indicates whether CUDA graph is to be applied.

For example:
```bash
./infer vit
```
Will execute ViT-S model without using CUDA graph, and

```bash
./infer regnet graph
```
Will execute RegNet-Y with CUDA graph applied.

Example output:

``` bash
$ ./infer vit
Deserialization required 155776 microseconds.
Total per-runner device persistent memory is 0
Total per-runner host persistent memory is 4816
Allocated activation device memory of size 3351040
CUDA lazy loading is enabled.
Input size: 602112bytes
Warming up! (5s)
Warmup complete. Starting inference. (30s)
inference complete. Dumping statistics:
Number of inference performed: 11517
Max inference time: 3.804ms
Min inference time: 2.58ms
Total inference time: 29989.5ms
Avg inference time: 2.60394ms
```
The last line of the output will inform you the end-to-end execution time of the DL model.
## 3.3 CUDA Runtime API calls

To evaluate how long CUDA runtime API calls take within a DL model, execute:

```bash
cd build/
nsys profile --trace=cuda,nvtx,cudnn,osrt,mpi --cuda-graph-trace=node --cudabacktrace=all ./infer {model_name} {graph}
nsys stats report{x}.nsys-rep | less
```

This will display information relevant to CUDA API calls. Specifically, search for CUDA API Summary.

Example output:

```
 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)               Name            
 --------  ---------------  ---------  ------------  ------------  ----------  -----------  ------------  ----------------------------
     26.7    7,950,915,560  1,526,192       5,209.6       4,730.0       3,760    7,945,584      16,969.3  cuLaunchKernel              
     14.9    4,446,411,668    738,480       6,021.0       5,383.0       4,697   64,931,953      76,507.2  cudaLaunchKernel                    
     10.3    3,059,976,404     12,312     248,536.1     263,482.5       2,968    8,021,950     103,294.8  cudaStreamSynchronize  
      6.9    2,063,562,353     24,619      83,819.9      78,265.0       4,425   68,473,825     472,592.4  cudaMemcpyAsync                      
```

Be aware that cuLaunchKernel is a different function to cudaLaunchKernel. The former is a function in CUDA driver, the latter one is the RuntimeAPI function.
"cudaStreamSynchronize" is called for every iteration of the DL model, therefore, the "Num Calls" represents the total number of inferences performed.
Using the number of cudaLaunchKernel and cudaMemcpyAsync, divided by number of cudaStreamSynchronize, you can obtain the average number of cudaLaunchKernel and
cudaMemcpyAsync within a DL model.

## 3.4 DL model launch time
Before beginning this section, make sure you have pandas installed. Run
```
python3 -m pip install pandas
```

Running the "nsys stats" command (cf. the previous section) will give you an sqlite file. Using that file, run

```
python3 model_launch_time.py report{x}.sqlite {--graph}
```
to obtain the average DL model launch time. The "--graph" argument should be passed if the relevant nsys stats is generated with CUDA graph enabled. Otherwise, this argument should not be used.

Example output:
```
$python3 model_launch_time.py report1.sqlite
Launch time: 151.02 us
```

## 3.5 CUDA Graph locking usage
Overall locking usage of CUDA functions can be obtained by executing:

```
LD_PRELOAD=shim.so ./infer_locking {model_name} {graph}
```

Example output:

```
$ LD_PRELOAD=shim.so ./infer_locking regnet graph
pthread_mutex_lock not initialized
Initialized pthread_mutex_lock
pthread_mutex_unlock not initialized
Initialized pthread_mutex_unlock
pthread_rwlock_wrlock not initialized
Initialized pthread_rwlock_wrlock
pthread_rwlock_unlock not initialized
Initialized pthread_rwlock_unlock
pthread_rwlock_rdlock not initialized
Initialized pthread_rwlock_rdlock
Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
Deserialization required 151338 microseconds.
Total per-runner device persistent memory is 187904
Total per-runner host persistent memory is 633024
Allocated activation device memory of size 19670528
CUDA lazy loading is enabled.
Input size: 602112bytes
"graph" specified, using cudaGraph
Warming up! (5s)
Warmup complete. Starting inference. (10 times)
inference complete. Dumping statistics:
Number of inference performed: 10
Max inference time: 6.008ms
Min inference time: 5.995ms
Total inference time: 60.017ms
Avg inference time: 6.0017ms
locking involved in invoking DL model: 171 locking operations
```

The last line of the output will give the total amount of locking operations (both mutex and RW lock) involved in calling the DL model.
This number does not include the cudaStreamSynchronize function.

## 3.6 GPU time

GPU time is already supplied in models/{model_name}.info.log (as GPU compute time), if the user wishes to confirm the GPU time, the user will need to follow the guidance in models/README.md. Please note that GPU compute time depends on the specific GPU used and may vary across different hardware/software.

