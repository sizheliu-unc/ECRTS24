This folder contains relevant material to evaluate deep learning inference, as well as CUDA graph.

### 0. Requirements
#### Hardware requirements
For this section you will need the following hardware:
1 x NVIDIA GPU (Pascal generation or later)
x86-64 hardware platform (ARM should work as well, but is not guaranteed)

#### Software requirements
Linux 5.4.0 or above
glibc 2.35 or above
CUDA Runtime 12.2.2
NVIDIA Driver 550.54.14
cuDNN 8.9.5.29
TensorRT 8.6.1.6
python 3.8

Both hardware and software will be supplied for the artifact evaluation.

#### Extra dependency
For users wishing to test on their own hardware/software, extra steps will be necessary.

First, the user must change the paths to CUDA and TensorRT in CMakeLists.txt and setup.sh according to the download location of these components.

The user will also need to generate DL model engine plans (see models/README.md), which are tied to the user's hardware and software and therefore non-portable.

### 1. Setup
To compile the relevant inference experiments, run:

``` bash
./setup.sh
```

Make sure you are in the cuda-infer/ folder. This will generate several executable files and .so files.

build/infer
build/infer_once
build/shim.so

### 2. End-to-end execution time

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
### 3. CUDA Runtime API calls

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

### 4. DL model launch time
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

### 5. CUDA Graph locking usage
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

#### 6. GPU time

GPU time is already supplied in models/{model_name}.info.log (as GPU compute time), if the user wishes to confirm the GPU time, the user will need to follow the guidance in models/README.md. Please note that GPU compute time depends on the specific GPU used and may vary across different hardware/software.
