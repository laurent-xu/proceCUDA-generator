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
      point_t _computeNormal(const point_t& p1, const point_t& p2,
                             const point_t& p3);
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
      bool is_inited_ = false;

    public:
      const std::vector <GLfloat> &getVertices() const { return _vertices; }
      const std::vector <GLuint> &getIndices() const { return _indices; }
      const std::vector <GLfloat> &getNormals() const { return _normals; }
      const std::vector <GLfloat> &getData() const { return _data; }

    public:
  };

}
