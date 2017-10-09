
set(TEST_DUAL_CONTOURING_SOURCES ${TEST_DUAL_CONTOURING_SOURCES}
        main.cc)

add_executable(test_dual_contouring ${TEST_DUAL_CONTOURING_SOURCES})
target_link_libraries(test_dual_contouring rendering)

set(${TESTS_LIST} test_dual_contouring PARENT_SCOPE)
