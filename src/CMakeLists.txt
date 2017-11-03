add_subdirectory(rendering)
add_subdirectory(density)
add_subdirectory(compilation)

add_library(rendering SHARED ${RENDERING_SOURCES})
add_library(density SHARED ${DENSITY_SOURCES})
add_library(compilation SHARED ${COMPILATION_SOURCES})
