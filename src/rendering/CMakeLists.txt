add_subdirectory(utils)
add_subdirectory(viewer)

set(RENDERING_SOURCES ${RENDERING_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/dual-contouring.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dual-contouring.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/qr-decomposition.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qr-decomposition.hpp
        PARENT_SCOPE)

