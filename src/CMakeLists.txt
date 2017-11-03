add_subdirectory(app)
add_subdirectory(compilation)
add_subdirectory(density)
add_subdirectory(rendering)

add_library(rendering SHARED ${RENDERING_SOURCES})
add_library(density SHARED ${DENSITY_SOURCES})
add_library(compilation SHARED ${COMPILATION_SOURCES})
add_library(emptygeneration SHARED ${EMPTY_GENERATION_SOURCES})


add_executable(app ${APP_SOURCES})
target_link_libraries(app; emptygeneration)
