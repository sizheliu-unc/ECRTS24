# ECRTS24
Supplemental material for ECRTS 2024 paper, *Autonomy Today: Many Delay-Prone Black Boxes*.

The supplemental material contains:
- README.md (current file)
- artifact
- glibc-2.38-phase-fair-boosting.patch
- cuda_lock_stats

## artifact
The *artifact* from this paper is located at the `artifact` folder, with a README.md for instructions. Note that the *artifact* located in this git repository may be subjected to changes by the authors for improvement. The artifact endorsed by the Artifact Evaluation Committee is available free of charge on the Dagstuhl Research Online Publication Server (DROPS).

## glibc RW lock phase-fair + priority boosting patch
The file glibc-2.38-phase-fair-boosting.patch can be applied to glibc-2.38 using
the following command:
```bash
$ git clone git://sourceware.org/git/glibc.git
$ cd glibc
$ git checkout release/2.38/master
$ git apply ${PATH}/glibc-2.38-phase-fair-boosting.patch
```


After this, to build the patched glibc, use:

```bash
$ $DESTDIR={}
$ $GLIBC_GIT={} 
$ mkdir $HOME/build/glibc
$ cd $HOME/build/glibc
$ $GLIBC_GIT/glibc/configure --prefix=/usr --disable-werror
$ make
$ make install DESTDIR=${DESTDIR}
```

To activate this patch for a C/C++ program, after compilation:
```bash
$ sudo apt-get -y install patchelf
$ patchelf --set-interpreter ${DESTDIR}/lib64/ld-linux-x86-64.so.2 --set-rpath ${DESTDIR}/lib64/ cc_program
$ ./cc_program
```


Note that the patch is only for experimental and demonstration purposes. The
pthread_rwlock in this patch can only work with scheduling policy SCHED_FIFO.

## CUDA locking statistics
cuda_lock_stats/ folder contains tools to generate statistics on locking usage in CUDA functions.

