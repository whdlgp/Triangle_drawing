cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project( cuda_triangle LANGUAGES CXX CUDA)

# source directory
set( SRCS cuda_triangle_kernel.cu cuda_triangle_main.cu image_writer.cpp )

# link
add_executable( ${CMAKE_PROJECT_NAME}.out ${SRCS} )
target_include_directories( ${CMAKE_PROJECT_NAME}.out PUBLIC )
target_link_libraries( ${CMAKE_PROJECT_NAME}.out )

