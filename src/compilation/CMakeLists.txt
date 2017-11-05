set(COMPILATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/AST.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/AST.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/CompileVisitor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CompileVisitor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/CheckVisitor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CheckVisitor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/DotVisitor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/DotVisitor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Visitor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.cc
    PARENT_SCOPE)

add_executable(compile2cuda main.cc)
target_link_libraries(compile2cuda; compilation)
