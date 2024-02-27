cmake_minimum_required(VERSION 3.27)
project(repasoexamen_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(repasoexamen_cuda main.cu)

set_target_properties(repasoexamen_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
