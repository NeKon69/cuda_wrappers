cmake_minimum_required(VERSION 3.20)

project(cuda_wrappers LANGUAGES CXX CUDA)

add_library(cuda_wrappers INTERFACE)

target_include_directories(cuda_wrappers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cuda_wrappers INTERFACE CUDA::cudart)

add_library(cuda_wrappers::cuda_wrappers ALIAS cuda_wrappers)
