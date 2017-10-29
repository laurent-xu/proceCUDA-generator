add_subdirectory(rendering)
add_subdirectory(density)

add_library(rendering SHARED ${RENDERING_SOURCES} ${DENSITY_SOURCES} ${NOISE_SOURCES})
