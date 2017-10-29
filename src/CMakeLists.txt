add_subdirectory(rendering)
add_subdirectory(density)
add_subdirectory(compilation)

add_library(rendering SHARED ${RENDERING_SOURCES}
                             ${DENSITY_SOURCES}
                             ${COMPILATION_SOURCES})
