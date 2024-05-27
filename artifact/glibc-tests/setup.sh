#!/bin/bash
set -e
if test -d glibc; then
    echo "glibc directory exists, skip cloning..."
else
    git clone --branch release/2.38/master --depth 1 https://sourceware.org/git/glibc.git
fi
cd glibc
git reset --hard origin/release/2.38/master
cd ..
rm -rf build dist bin trace
mkdir build dist bin trace
LD_LIBRARY_PATH=""
GLIBC_PATH=$(realpath ./glibc)
BUILD_PATH=$(realpath ./build)
DIST_PATH=$(realpath ./dist)
cd glibc 
git apply ../glibc-2.38-phase-fair-boosting.patch
cd $BUILD_PATH
echo $LD_LIBRARY_PATH
$GLIBC_PATH/configure --prefix=/usr --disable-werror
make -j32
make install DESTDIR=$DIST_PATH
