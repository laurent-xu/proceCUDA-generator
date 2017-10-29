add_subdirectory(rendering)
add_subdirectory(density)
add_subdirectory(noise)

add_library(rendering SHARED ${RENDERING_SOURCES} ${DENSITY_SOURCES} ${NOISE_SOURCES})
