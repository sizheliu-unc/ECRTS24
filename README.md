# ECRTS24
Supplemental material for ECRTS24 paper: Autonomy Today: Many Delay-Prone Black Boxes.

The supplemental material contains:
- README.md (current file)
- glibc-2.38-phase-fair-boosting.patch
- cuda_lock_stats

## glibc RW lock phase-fair + priority boosting patch
The file glibc-2.38-phase-fair-boosting.patch can be applied to glibc-2.38 using
the following command:

> $ git clone git://sourceware.org/git/glibc.git
> $ cd glibc
> $ git checkout release/2.38/master
> $ git apply ${PATH}/glibc-2.38-phase-fair-boosting.patch

After this, to build the patched glibc, use:

> $ $DESTDIR={} # Fill in the installation location
> $ $GLIBC_GIT={} # Fill in the glibc git folder location
> $ mkdir $HOME/build/glibc
> $ cd $HOME/build/glibc
> $ $GLIBC_GIT/glibc/configure --prefix=/usr --disable-werror
> $ make
> $ make install DESTDIR=${DESTDIR}

To activate this patch for a C/C++ program, after compilation:
> $ sudo apt-get -y install patchelf
> $ patchelf --set-interpreter ${DESTDIR}/lib64/ld-linux-x86-64.so.2 --set-rpath ${DESTDIR}/lib64/ cc_program
> $ ./cc_program

Note that the patch is only for experimental and demonstration purposes. The
pthread_rwlock in this patch can only work with scheduling policy SCHED_FIFO.

## CUDA locking statistics
cuda_lock_stats/ folder contains tools to generate statistics on locking usage in CUDA functions.

