cmake_minimum_required( VERSION 3.0 )
project( cpu_triangle )

# source directory
set( SRCS )
aux_source_directory( "." SRCS )

# defendancy
find_package( Threads )
find_package(OpenMP REQUIRED)

# link
add_executable( ${CMAKE_PROJECT_NAME}.out ${SRCS} )
target_include_directories( ${CMAKE_PROJECT_NAME}.out PUBLIC  )
target_link_libraries( ${CMAKE_PROJECT_NAME}.out ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX )

