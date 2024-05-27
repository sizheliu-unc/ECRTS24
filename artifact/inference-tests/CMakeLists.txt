cmake_minimum_required(VERSION 3.0.2)
project(inference LANGUAGES CXX CUDA)

find_package(CUDA 12.2 REQUIRED)
enable_language(CUDA)

set(CUDA_PATH /usr/local/cuda-12.2.2)
set(TENSORRT_PATH /usr/local/TensorRT-8.6.1.6)

include_directories(./src ${TENSORRT_PATH}/include ${CUDA_PATH}/include)


add_library(util src/util.cpp)
add_library(dummy SHARED IMPORTED)
set_target_properties(dummy PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/libdummy.so)
add_executable(infer src/infer.cu)
add_executable(infer_locking src/infer_locking.cu)
# add_executable(saxpy src/saxpy.cu)

target_compile_options(infer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -lcublas
    -lm
    --std=c++17
    >)
target_link_libraries(infer util ${CUDA_LIBRARIES} ${TENSORRT_PATH}/lib/libnvinfer.so ${CUDA_PATH}/lib64/libcudart.so)

target_compile_options(infer_locking PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -lcublas
    -lm
    --std=c++17
    -O0
    >)
target_link_libraries(infer_locking util ${CUDA_LIBRARIES} ${TENSORRT_PATH}/lib/libnvinfer.so ${CUDA_PATH}/lib64/libcudart.so dummy)

# target_compile_options(saxpy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#     -lcudart
#     -lcublas
#     -lm
#     --std=c++17
#     -I /home/wagle/kutrace>)
# # TARGET_LINK_LIBRARIES(infer imgproc ${CUDA_LIBRARIES} ${TENSORRT_PATH}/lib/libnvinfer.so)
# target_link_libraries(saxpy kutrace ${CUDA_LIBRARIES} ${CUDA_PATH}/lib64/libcudart.so)
