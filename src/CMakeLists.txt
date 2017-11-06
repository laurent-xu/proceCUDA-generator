add_subdirectory(app)
add_subdirectory(compilation)
add_subdirectory(density)
add_subdirectory(rendering)

add_library(rendering SHARED ${RENDERING_SOURCES})
add_library(compilation SHARED ${COMPILATION_SOURCES})

set(APP_SOURCES ${APP_SOURCES}
    PARENT_SCOPE)

  set(CUDA_APP_SOURCES ${CUDA_APP_SOURCES}
    PARENT_SCOPE)
