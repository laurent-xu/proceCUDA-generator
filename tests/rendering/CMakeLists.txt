set(TEST_DUAL_CONTOURING_SOURCES ${TEST_DUAL_CONTOURING_SOURCES}
        main.cc)

# Detect and add SFML
find_package(SFML 2 REQUIRED system window graphics network audio)
if (!SFML_FOUND)
    MESSAGE("SFML NOT FOUND")
endif()
include_directories(${SFML_INCLUDE_DIRS})

add_executable(test_dual_contouring ${TEST_DUAL_CONTOURING_SOURCES})
target_link_libraries(test_dual_contouring; rendering; ${SFML_LIBRARIES}; ${GLEW_LIBRARIES}; GL;)

set(${TESTS_LIST} test_dual_contouring PARENT_SCOPE)
