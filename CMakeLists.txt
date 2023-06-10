cmake_minimum_required(VERSION 3.18)
project(GoL_CUDA CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(main glad/glad.o main.cu gpu.cuh)
target_link_libraries(main GL GLU glfw CUDA::cudart_static dl)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(main PRIVATE "-Wall -Wextra -O2")
