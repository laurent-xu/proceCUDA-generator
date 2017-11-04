add_subdirectory(utils)
add_subdirectory(viewer)

set(RENDERING_SOURCES ${RENDERING_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/hermitian-grid.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hermitian-grid.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/qr-decomposition.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qr-decomposition.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vertices-grid.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vertices-grid.cpp
        PARENT_SCOPE)

