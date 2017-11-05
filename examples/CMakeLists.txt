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
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples)

  if(EXISTS ${GENERATED_FILES}/${current_file}.cu)
    cuda_add_library(cudalib${current_file} SHARED ${GENERATED_FILES}/${current_file}.cu)
    add_library(lib${current_file} SHARED ${GENERATED_FILES}/${current_file}.cc)
    add_custom_target(${current_target}lib ALL DEPENDS cudalib${current_file} lib${current_file} ${current_target})
  endif()
endforeach()
