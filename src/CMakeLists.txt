add_subdirectory(app)
add_subdirectory(compilation)
add_subdirectory(density)
add_subdirectory(rendering)

add_library(rendering SHARED ${RENDERING_SOURCES})
add_library(compilation SHARED ${COMPILATION_SOURCES})
add_library(emptygeneration SHARED ${EMPTY_GENERATION_SOURCES})
#CUDA_ADD_LIBRARY(cudaemptygeneration SHARED ${CUDA_EMPTY_GENERATION_SOURCES})

add_executable(app ${APP_SOURCES})
target_link_libraries(app; emptygeneration; rendering; ${SFML_LIBRARIES}; ${GLEW_LIBRARIES}; GL;)
#target_link_libraries(app; emptygeneration)

cuda_add_executable(cudaapp ${CUDA_APP_SOURCES})
target_link_libraries(cudaapp; cudaemptygeneration)
