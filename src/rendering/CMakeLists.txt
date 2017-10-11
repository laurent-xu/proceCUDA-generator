add_subdirectory(utils)

set(RENDERING_SOURCES ${RENDERING_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/dual-contouring.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dual-contouring.hh
        PARENT_SCOPE)

# Detect and add GLEW
find_package(GLEW REQUIRED)
if (!GLEW_FOUND)
    MESSAGE("GLEW NOT FOUND")
endif()
include_directories(${GLEW_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
if (!OPENGL_FOUND)
    MESSAGE("OPENGL NOT FOUND")
endif()
include_directories(${GLEW_INCLUDE_DIRS})
