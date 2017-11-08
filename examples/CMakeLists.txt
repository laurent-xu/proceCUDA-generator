file(GLOB files "*.proc")
foreach(file ${files})
  get_filename_component(current_file ${file} NAME_WE)
  set(current_target GEN${current_file})
  add_custom_target(${current_target} ALL DEPENDS make_dot compile2cuda)
  add_custom_command(TARGET ${current_target}
                     COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/make_dot
                                ${file} ${GENERATED_FILES}/${current_file}.dot
                                2>/dev/null \;
                             if [ -f ${GENERATED_FILES}/${current_file}.dot ] \;
                             then
                               dot -Tsvg ${GENERATED_FILES}/${current_file}.dot
                               > ${GENERATED_FILES}/${current_file}.svg \;
                             fi \;
                             ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/compile2cuda
                                ${file} ${GENERATED_FILES}/${current_file}.cu
                                ${GENERATED_FILES}/${current_file}.cc
                                2>/dev/null
                             \; echo ok
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples
                     DEPENDS ${file})

  if(EXISTS ${GENERATED_FILES}/${current_file}.cu)
    cuda_add_executable(cudaapp${current_file} ${CUDA_APP_SOURCES} ${GENERATED_FILES}/${current_file}.cu)
    target_link_libraries(cudaapp${current_file} ; rendering; ${SFML_LIBRARIES}; ${GLEW_LIBRARIES}; GL;)
    add_executable(app${current_file} ${APP_SOURCES} ${GENERATED_FILES}/${current_file}.cc)
    target_link_libraries(app${current_file}; rendering; ${SFML_LIBRARIES}; ${GLEW_LIBRARIES}; GL;)
    set(GENERATED_TARGETS ${GENERATED_TARGETS} ${current_target})
  endif()
endforeach()
add_custom_target(gen_examples DEPENDS ${GENERATED_TARGETS})
add_custom_command(TARGET gen_examples
                   COMMAND cmake . && make
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
