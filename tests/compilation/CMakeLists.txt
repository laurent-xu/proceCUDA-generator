set(TEST_COMPILATION_SOURCES ${TEST_COMPILATION_SOURCES}
        main.cc)

add_executable(test_compilation ${TEST_COMPILATION_SOURCES}
                                ${COMPILATION_SOURCES})

target_link_libraries(test_compilation; compilation)

set(${TESTS_LIST} test_compilation PARENT_SCOPE)
