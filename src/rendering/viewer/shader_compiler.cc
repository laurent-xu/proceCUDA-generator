//
// Created by leo on 8/10/16.
//

#include "shader_compiler.hh"
#include "file_reader.hh"

#include <iostream>
// GLEW
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>

GLuint shader_compiler::compile(std::string filename, GLenum shader_type) {
  std::string str = file_reader::readFile(filename);
  const GLchar *vertexShaderSource = str.c_str();
  GLuint shader;
  shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, &vertexShaderSource, NULL);
  // Checking for compilation errors
  GLint success;
  GLchar infolog[512];
  int bufflen = 0;
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &bufflen);
  if (success != GL_TRUE) {
    glGetShaderInfoLog(shader, 512, NULL, infolog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
    << infolog << std::endl;
    return 0;
  }
  return shader;
}

