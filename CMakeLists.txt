cmake_minimum_required(VERSION 3.0)

project(ProceCUDA)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

find_package(CUDA REQUIRED)
find_package (Threads REQUIRED)

set(CUDA_NVCC_FLAGS_TMP -Xcompiler
    -D__CORRECT_ISO_CPP11_MATH_H_PROTO
        -D_MWAITXINTRIN_H_INCLUDED
        -D_FORCE_INLINES
        -D__STRICT_ANSI__
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -fPIC --expt-relaxed-constexpr
    -Wno-deprecated-gpu-targets)

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

execute_process(COMMAND git submodule update --init -- ThirdParties/glm
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(GLM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glm
    CACHE PATH "glm include directory")
include_directories(${GLM_INCLUDE_DIRS})

execute_process(COMMAND git submodule update --init -- ThirdParties/glfw
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
execute_process(COMMAND cmake -DBUILD_SHARED_LIBS=ON .
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glfw)
set(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glfw/include
    CACHE PATH "glfw include directory")
include_directories(${GLFW_INCLUDE_DIRS})
set(GLFW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/glfw/src/libglfw.so.3
  CACHE PATH "glew lib directory")

find_package(OpenGL REQUIRED)
if (!OPENGL_FOUND)
    MESSAGE("OPENGL NOT FOUND")
endif()

# Detect and add SFML
MESSAGE("SFML NOT FOUND")
execute_process(COMMAND git submodule update --init -- ThirdParties/SFML
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
execute_process(COMMAND cmake .
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML)
execute_process(COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML)
set(SFML_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML/include
  CACHE PATH "SFML include directory")
set(SFML_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML/lib/libsfml-system.so;
                 ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML/lib/libsfml-window.so;
                 ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML/lib/libsfml-graphics.so;
                 ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML/lib/libsfml-network.so;
                 ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/SFML/lib/libsfml-audio.so
    CACHE PATH "SFML lib directory")
include_directories(${SFML_INCLUDE_DIRS})

# C++14 and extra flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wextra -std=c++11 -O3 -DDEBUG -fopenmp")

# Sets library output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(GENERATED_FILES ${CMAKE_CURRENT_SOURCE_DIR}/generated_files)
execute_process(COMMAND mkdir -p ${GENERATED_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
)

# Adds source directory
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
