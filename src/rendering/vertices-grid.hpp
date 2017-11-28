//
// Created by leo on 11/4/17.
//

#pragma once

#include <GL/glew.h>
#include "hermitian-grid.hh"

namespace rendering {

  class VerticesGrid {
    public:
      VerticesGrid(const HermitianGrid &hermitianGrid, float scale);

    public:
      void draw();
      void initVAO();

    private:
      void computeVBO(const HermitianGrid& hermitianGrid, float scale);
      void VBO_kernel(const HermitianGrid& hermitianGrid, float scale,
                      size_t x, size_t y, size_t z, size_t& vbo_idx);
      void _addVertex(point_t vertex, std::vector<GLfloat>& buffer_vect);

    private:
      std::vector <GLfloat> _vertices;
      std::vector <GLuint> _indices;
      std::vector <GLfloat> _normals;
      std::vector <GLfloat> _data;
      GLuint _VAO;
      GLuint _VBO;
      GLuint _EBO;

    public:
      const std::vector <GLfloat> &getVertices() const { return _vertices; }
      const std::vector <GLuint> &getIndices() const { return _indices; }
      const std::vector <GLfloat> &getNormals() const { return _normals; }
      const std::vector <GLfloat> &getData() const { return _data; }

    public:
  };

}
