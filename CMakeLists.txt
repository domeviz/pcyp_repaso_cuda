cmake_minimum_required(VERSION 3.27)
project(repasoexamen_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)
#find_package(fmt)

add_executable(repasoexamen_cuda main.cu)

target_include_directories(repasoexamen_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#target_link_libraries(repasoexamen_cuda fmt::fmt ${CUDART_LIBRARY})

set_target_properties(repasoexamen_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
