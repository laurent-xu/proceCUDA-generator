//
// Created by leo on 8/10/16.
//

#ifndef TUTO1_SHADER_COMPILER_HH
#define TUTO1_SHADER_COMPILER_HH

#include <string>
#include <GL/glew.h>

class shader_compiler {
  public:
  static GLuint compile(std::string filename, GLenum shader_type);

};


#endif //TUTO1_SHADER_COMPILER_HH
