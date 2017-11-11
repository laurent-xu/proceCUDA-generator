set(APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/generation_kernel.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rendering.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/rendering.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/make_density.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/make_density.cc
    PARENT_SCOPE)

set(CUDA_APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/generation_kernel.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rendering.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/rendering.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/make_density.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/make_density.cu
    PARENT_SCOPE)
