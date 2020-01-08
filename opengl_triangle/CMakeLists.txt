cmake_minimum_required( VERSION 3.0 )
project( opengl_triangle )

# source directory
set( SRCS )
aux_source_directory( "." SRCS )

# defendancy
find_package( Threads )
find_package( glfw3 3.3 REQUIRED )
find_package( OpenGL REQUIRED )
set( GLAD_DIR /usr/local/glad )

# link
add_executable( ${CMAKE_PROJECT_NAME}.out ${SRCS} )
target_include_directories( ${CMAKE_PROJECT_NAME}.out PUBLIC ${OPENGL_INCLUDE_DIR} ${GLAD_DIR}/include )
target_link_libraries( ${CMAKE_PROJECT_NAME}.out ${CMAKE_THREAD_LIBS_INIT} glfw ${OPENGL_gl_LIBRARY} glad )

