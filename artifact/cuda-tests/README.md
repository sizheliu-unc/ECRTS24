# CUDA Tests
# Test 1: CUDA Kernel Launch-Memcpy Contention Test
This test demonstrates the contention that can occur between pthread rwlocks and mutexes when many CUDA memcopies and CUDA kernels are launched in parallel.
This test spawns 50 pthreads. Of the 50 threads, 25 will perform 20 CUDA kernel launches, and the other 25 threads perform 20 Host-to-Device memcopies and 20 Device-to-Host memcopies.

### Building code
First choose an NVIDIA Driver version in the `artifact/nvidia_drivers`.\
Install a driver with `sudo sh ./<driverfile>.run --silent`.

Enter the  `artifact/cuda-tests/cuda_launch_kernel_memcpy_contention` directory.
To build the test for CUDA Toolkit 11.2 + NVIDIA Driver 460.27.04, run ```make cuda11```.\
To build the test for CUDA Toolkit 12.2 + NVIDIA Driver {550.54.14, 525.89.02}, run ```make cuda12```.

### Running code
To run the test and post-process the output for viewing, run ```make run```\
This will produce an html file called ```cuda_launch_kernel_memcpy_contention.trace.trimmed.html```\
This html file can be copied back to your host machine via WinSCP, FileZilla, Cyberduck, etc, and opened in any web browser.

### Viewing the trace output
The html file will be a bit big, so it might take a minute to load in the browser.

We use the KUtrace tracing framework written by Dr. Richard L. Sites.\
The full source code for KUtrace can be found here: https://github.com/dicksites/KUtrace

### Reading KUtrace traces
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
For reference, the actual functions in our code that drop the markers are: kutrace::mark_a, kutrace::mark_b, kutrace::mark_c, where a,b,c just tell it what color to make the marker flag.

<video src="cuda_launch_kernel_memcpy_contention/kutrace_demo_small.mp4" height="480" controls></video>

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

# Test 2: CUDA Launch Kernel Time
This test will launch 100 Saxpy CUDA kernels, time each launch, and print out statistics on the 100 launch times.

## Building code
First choose an NVIDIA Driver version in the `artifact/nvidia_drivers`.\
Install a driver with `sudo sh ./<driverfile>.run --silent`.

Enter the  `artifact/cuda-tests/cuda_launch_kernel_time` directory.
To build the test for CUDA Toolkit 11.2 + NVIDIA Driver 460.27.04, run ```make cuda11```.\
To build the test for CUDA Toolkit 12.2 + NVIDIA Driver {550.54.14, 525.89.02}, run ```make cuda12```.

## Running code
To run the test and post-process the output for viewing, run ```make run```

## What you should see in the output
The program will print out the statistics of the CUDA kernel launch times.
It also generates a KUtrace html file, `cuda_launch_kernel_time.trace.trimmed.html` that you can open and view the same as in Test 1.

# Test 3: CUDA Memcpy Time
This test will launch 100 asynchronous CUDA Device-to-Host and Host-to-Device memcopies, time each async copy's launch time, and print out statistics on the 100 launch times.

## Building code
First choose an NVIDIA Driver version in the `artifact/nvidia_drivers`.\
Install a driver with `sudo sh ./<driverfile>.run --silent`.

Enter the  `artifact/cuda-tests/cuda_memcpy_time` directory.
To build the test for CUDA Toolkit 11.2 + NVIDIA Driver 460.27.04, run ```make cuda11```.\
To build the test for CUDA Toolkit 12.2 + NVIDIA Driver {550.54.14, 525.89.02}, run ```make cuda12```.

## Running code
To run the test and post-process the output for viewing, run ```make run```

## What you should see in the output
The program will print out the statistics of the CUDA kernel launch times.
It also generates a KUtrace html file, `cuda_memcpy_time.trace.trimmed.html` that you can open and view the same as in Test 1.


# Troubleshooting
If you run into any issues testing or installing the drivers, please email wagle@cs.unc.edu