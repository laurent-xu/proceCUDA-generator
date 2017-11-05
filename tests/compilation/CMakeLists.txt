set(TEST_COMPILATION_SOURCES main.cc)

add_executable(make_dot ${TEST_COMPILATION_SOURCES})

target_link_libraries(make_dot; compilation)

set(${TESTS_LIST} make_dot PARENT_SCOPE)
