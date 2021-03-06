//
// Created by leo on 11/4/17.
//

#include <utils/cudamacro.hh>
#include <iostream>
#include "vertices-grid.hpp"

namespace rendering
{
VerticesGrid::VerticesGrid(const HermitianGrid &hermitianGrid, float scale)
{
  computeVBO(hermitianGrid, scale);
  // initVAO();
}

void VerticesGrid::VBO_kernel(const HermitianGrid &hermitianGrid, float scale,
                              size_t x, size_t y, size_t z, size_t& vbo_idx)
{
  auto &node = hermitianGrid.getValueAt(x, y, z);
  auto dimensions = hermitianGrid.getDimensions();

  if (hermitianGrid.pointContainsFeature(x, y, z)) {
    for (auto var_x: {0, 1})
      for (auto var_y: {0, 1})
        for (auto var_z: {0, 1}) {
          auto new_x = var_x + x;
          auto new_y = var_y + y;
          auto new_z = var_z + z;
          auto x1 = new_x;
          auto y1 = new_x == x ? new_y : y;
          auto z1 = z;
          auto x2 = x;
          auto y2 = new_z == z ? new_y : y;
          auto z2 = new_z;
          if (var_x + var_y + var_z != 2 || new_x >= dimensions.x ||
              new_y >= dimensions.y || new_z >= dimensions.z)
            continue;
          if (hermitianGrid.pointContainsFeature(x1, y1, z1) &&
              hermitianGrid.pointContainsFeature(x2, y2, z2) &&
              hermitianGrid.pointContainsFeature(new_x, new_y, new_z)) {

            auto point1 =
                    hermitianGrid.getValueAt(x, y, z);
            auto point2 =
                    hermitianGrid.getValueAt(x1, y1, z1);
            auto point3 =
                    hermitianGrid.getValueAt(new_x, new_y,
                                             new_z);
            auto point4 =
                    hermitianGrid.getValueAt(x2, y2, z2);

            // auto normal = _computeNormal(point1, point2, point3);
            for (auto p: {point1, point2, point3, point4}) {
              _addVertex(p.normal.scale(1.0 / p.normal.norm()), _normals);
              _addVertex(p.vertex_pos.scale(scale), _vertices);
            }
            for (size_t i: {0, 1, 2, 3, 2, 0})
              _indices.push_back(vbo_idx + i);
            vbo_idx += 4;
          }
        }
  }
}

void VerticesGrid::computeVBO(const HermitianGrid &hermitianGrid, float scale)
{
  size_t vbo_idx = 0;
  // Indices of previous vertices to avoid duplication.
  auto dimensions = hermitianGrid.getDimensions();
  for (int z = 0; z < dimensions.z; z++)
    for (int y = 0; y < dimensions.y; y++)
      for (int x = 0; x < dimensions.x; x++)
        VBO_kernel(hermitianGrid, scale, x, y, z, vbo_idx);

  CERR << "normals " << _normals.size() << std::endl;
  CERR << "vertices " << _vertices.size() << std::endl;
  for (size_t i = 0; i < _vertices.size(); i += 3)
  {
    for (size_t j: {0, 1, 2})
      _data.push_back(_vertices[i + j]);
    for (size_t j: {0, 1, 2})
      _data.push_back(_normals[i + j]);
  }
}

void VerticesGrid::_addVertex(point_t vertex, std::vector<GLfloat> &buffer_vect)
{
  buffer_vect.push_back(vertex.x);
  buffer_vect.push_back(vertex.y);
  buffer_vect.push_back(vertex.z);
}

void VerticesGrid::initVAO()
{
  auto &data_vect = _data;
  GLfloat *data = &data_vect[0];

  auto &indices_vect = _indices;
  GLuint *indices = &indices_vect[0];

  glGenBuffers(1, &_EBO);
  glGenVertexArrays(1, &_VAO);
  glGenBuffers(1, &_VBO);
  glBindVertexArray(_VAO);
  {
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER, _data.size() * sizeof (GLfloat), data,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indices.size() * sizeof (GLuint),
                 indices, GL_STATIC_DRAW);
    // vertices
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat),
                          (GLvoid *) (0));
    glEnableVertexAttribArray(0);
    // normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat),
                          (GLvoid *) (3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
  }
  glBindVertexArray(0);
}

void VerticesGrid::draw()
{
  glBindVertexArray(_VAO);
  {
    glDrawElements(GL_TRIANGLES, (GLsizei) _indices.size(), GL_UNSIGNED_INT, 0);
  }
  glBindVertexArray(0);
}
}
