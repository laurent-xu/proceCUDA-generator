add_subdirectory(compilation)
add_subdirectory(rendering)

add_custom_target(samples DEPENDS ${TESTS_LIST})
