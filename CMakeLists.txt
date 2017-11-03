cmake_minimum_required(VERSION 3.0)

project(ProceCUDA)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

MESSAGE("glew not found, installing glew")
execute_process(COMMAND git submodule update --init -- ThirdParties/glew
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
execute_process(COMMAND make
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glew/auto)
execute_process(COMMAND make
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glew)
set(GLEW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glew/include
    CACHE PATH "glew include directory")
set(GLEW_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glew/lib/libGLEW.so
  CACHE PATH "glew lib directory")
set(GLEW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glew/include
    CACHE PATH "glew include directory")
set(GLEW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glew/lib/libGLEW.so
  CACHE PATH "glew lib directory")
include_directories(${GLEW_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
if (!OPENGL_FOUND)
    MESSAGE("OPENGL NOT FOUND")
endif()
include_directories(${GLEW_INCLUDE_DIRS})

# C++14 and extra flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wextra -std=c++1y -O0")

# Sets library output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src)

include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
)

# Adds source directory
add_subdirectory(src)
add_subdirectory(tests)
