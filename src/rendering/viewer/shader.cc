//
// Created by leo on 8/14/16.
//

#include "shader.hh"

Shader::Shader(const GLchar *vertexSourcePath,
               const GLchar *fragmentSourcePath) {
  std::string vertexCode = file_reader::readFile(vertexSourcePath);
  std::string fragmentCode = file_reader::readFile(fragmentSourcePath);
  const GLchar *vShaderCode = vertexCode.c_str();
  const GLchar *fShaderCode = fragmentCode.c_str();
  GLuint vertex, fragment;
  compileShader(vertex, vShaderCode, GL_VERTEX_SHADER);
  compileShader(fragment, fShaderCode, GL_FRAGMENT_SHADER);
  this->Program = glCreateProgram();
  glAttachShader(this->Program, vertex);
  glAttachShader(this->Program, fragment);
  glLinkProgram(this->Program);
  GLint success;
  GLchar infoLog[512];
  glGetProgramiv(this->Program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(this->Program, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
    << infoLog << std::endl;
  }
  glDeleteShader(vertex);
  glDeleteShader(fragment);
}

void Shader::compileShader(GLuint &shader, const GLchar *shaderCode,
                           GLint shaderType) {
  GLint success;
  GLchar infolog[512];
  shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, &shaderCode, NULL);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(shader, 512, NULL, infolog);
    std::cout << "ERROR::PROGRAM::COMPILE::COMPILATION_FAILED\n"
    << infolog << std::endl;
  }
}

void Shader::Use() {
  glUseProgram(this->Program);
}

GLuint &Shader::getProgram() {
  return Program;
}


