//
// Created by leo on 8/14/16.
//

#ifndef OPENGL_TUTORIALS_SHADER_HH
#define OPENGL_TUTORIALS_SHADER_HH

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GL/glew.h>

#include "file_reader.hh"

class Shader {
  public:
    GLuint Program;
    Shader(const GLchar *vertexSourcePath, const GLchar *fragmentSourcePath);
    void Use();
    GLuint &getProgram();
    static void compileShader(GLuint &shader, const GLchar *shaderCode, GLint shaderType);
};


#endif //OPENGL_TUTORIALS_SHADER_HH
