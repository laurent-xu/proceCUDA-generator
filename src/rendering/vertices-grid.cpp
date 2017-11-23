//
// Created by leo on 11/4/17.
//

#include <utils/cudamacro.hh>
#include <iostream>
#include "vertices-grid.hpp"

namespace rendering {

  VerticesGrid::VerticesGrid(const HermitianGrid &hermitianGrid, float scale) {
    computeVBO(hermitianGrid, scale);
    initVAO();
  }

  point_t VerticesGrid::_computeNormal(const point_t &p1, const point_t &p2, const point_t &p3) {
    auto U = p2 - p1;
    auto V = p3 - p1;
    float x = (float) (U.y * V.z - U.z * V.y);
    float y = (float) (U.z * V.x - U.x * V.z);
    float z = (float) (U.x * V.y - U.y * V.x);
    return point_t(x, y, z);
  }

  void VerticesGrid::computeVBO(const HermitianGrid &hermitianGrid, float scale) {
    unsigned int vbo_idx = 0;
    // Indices of previous vertices to avoid duplication.
    for (int z = 0; z < hermitianGrid.getDimensions().z; z++) {
      for (int y = 0; y < hermitianGrid.getDimensions().y; y++) {
        for (int x = 0; x < hermitianGrid.getDimensions().x; x++) {
          auto &node = hermitianGrid.getValueAt(x, y, z);
          if (hermitianGrid.pointContainsFeature(x, y, z)) {
            if (x + 1 < hermitianGrid.getDimensions().x && hermitianGrid.pointContainsFeature(x + 1, y, z)
                && y + 1 < hermitianGrid.getDimensions().y && hermitianGrid.pointContainsFeature(x, y + 1, z)
                && hermitianGrid.pointContainsFeature(x + 1, y + 1, z)) {
              auto normal = _computeNormal(hermitianGrid.getValueAt(x, y, z).vertex_pos.scale(scale),
                                           hermitianGrid.getValueAt(x + 1, y, z).vertex_pos.scale(scale),
                                           hermitianGrid.getValueAt(x + 1, y + 1, z).vertex_pos.scale(scale));
              normal = normal.scale(1 / normal.norm());
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(hermitianGrid.getValueAt(x, y, z).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x + 1, y, z).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x + 1, y + 1, z).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x, y + 1, z).vertex_pos.scale(scale), _vertices);
              _indices.push_back(vbo_idx);
              _indices.push_back(vbo_idx + 1);
              _indices.push_back(vbo_idx + 2);
              _indices.push_back(vbo_idx);
              _indices.push_back(vbo_idx + 2);
              _indices.push_back(vbo_idx + 3);
              vbo_idx += 4;
            }
            if (x + 1 < hermitianGrid.getDimensions().x && hermitianGrid.pointContainsFeature(x + 1, y, z)
                && z + 1 < hermitianGrid.getDimensions().z && hermitianGrid.pointContainsFeature(x, y, z + 1)
                && hermitianGrid.pointContainsFeature(x + 1, y, z + 1)) {
              auto normal = _computeNormal(hermitianGrid.getValueAt(x, y, z).vertex_pos.scale(scale),
                                           hermitianGrid.getValueAt(x + 1, y, z).vertex_pos.scale(scale),
                                           hermitianGrid.getValueAt(x + 1, y, z + 1).vertex_pos.scale(scale));
              normal = normal.scale(1 / normal.norm());
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(hermitianGrid.getValueAt(x, y, z).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x + 1, y, z).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x + 1, y, z + 1).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x, y, z + 1).vertex_pos.scale(scale), _vertices);
              _indices.push_back(vbo_idx);
              _indices.push_back(vbo_idx + 1);
              _indices.push_back(vbo_idx + 2);
              _indices.push_back(vbo_idx);
              _indices.push_back(vbo_idx + 2);
              _indices.push_back(vbo_idx + 3);
              vbo_idx += 4;
            }
            if (y + 1 < hermitianGrid.getDimensions().y && hermitianGrid.pointContainsFeature(x, y + 1, z)
                && z + 1 < hermitianGrid.getDimensions().z && hermitianGrid.pointContainsFeature(x, y, z + 1)
                && hermitianGrid.pointContainsFeature(x, y + 1, z + 1)) {
              auto normal = _computeNormal(hermitianGrid.getValueAt(x, y, z).vertex_pos.scale(scale),
                                           hermitianGrid.getValueAt(x, y + 1, z).vertex_pos.scale(scale),
                                           hermitianGrid.getValueAt(x, y + 1, z + 1).vertex_pos.scale(scale));
              normal = normal.scale(1 / normal.norm());
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(normal, _normals);
              _addVertex(hermitianGrid.getValueAt(x, y, z).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x, y + 1, z).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x, y + 1, z + 1).vertex_pos.scale(scale), _vertices);
              _addVertex(hermitianGrid.getValueAt(x, y, z + 1).vertex_pos.scale(scale), _vertices);
              _indices.push_back(vbo_idx);
              _indices.push_back(vbo_idx + 1);
              _indices.push_back(vbo_idx + 2);
              _indices.push_back(vbo_idx);
              _indices.push_back(vbo_idx + 2);
              _indices.push_back(vbo_idx + 3);
              vbo_idx += 4;
            }
          }
        }
      }
    }
    CERR << "normals " << _normals.size() << std::endl;
    CERR << "vertices " << _vertices.size() << std::endl;
    for (size_t i = 0; i < _vertices.size(); i += 3) {
      _data.push_back(_vertices[i]);
      _data.push_back(_vertices[i + 1]);
      _data.push_back(_vertices[i + 2]);
      _data.push_back(_normals[i]);
      _data.push_back(_normals[i + 1]);
      _data.push_back(_normals[i + 2]);
    }
  }

  void VerticesGrid::_addVertex(point_t vertex, std::vector<GLfloat> &buffer_vect) {
    buffer_vect.push_back(vertex.x);
    buffer_vect.push_back(vertex.y);
    buffer_vect.push_back(vertex.z);
  }

  void VerticesGrid::initVAO() {
    auto &data_vect = _data;
    GLfloat *data = &data_vect[0];

    auto &indices_vect = _indices;
    GLuint *indices = &indices_vect[0];

    glGenBuffers(1, &_EBO);
    glGenVertexArrays(1, &_VAO);
    glGenBuffers(1, &_VBO);
    glBindVertexArray(_VAO); {
      glBindBuffer(GL_ARRAY_BUFFER, _VBO);
      glBufferData(GL_ARRAY_BUFFER, _data.size() * sizeof (GLfloat), data, GL_STATIC_DRAW);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indices.size() * sizeof (GLuint), indices, GL_STATIC_DRAW);
      // vertices
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid *) (0));
      glEnableVertexAttribArray(0);
      // normals
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid *) (3 * sizeof(GLfloat)));
      glEnableVertexAttribArray(1);
    } glBindVertexArray(0);
  }

  void VerticesGrid::draw() {
    std::cout << _indices.size() << std::endl;
    glBindVertexArray(_VAO); {
      glDrawElements(GL_TRIANGLES, (GLsizei) _indices.size(), GL_UNSIGNED_INT, 0);
    } glBindVertexArray(0);
  }

  void VerticesGrid::addCube(const HermitianGrid &hermitianGrid, unsigned int &vbo_idx, float scale,
                             point_t a, point_t b, point_t c, point_t d)
  {
    if (a.x + 1 < hermitianGrid.getDimensions().x && hermitianGrid.pointContainsFeature(a.x + 1, a.y, a.z)
        && a.y + 1 < hermitianGrid.getDimensions().y && hermitianGrid.pointContainsFeature(a.x, a.y + 1, a.z)
        && hermitianGrid.pointContainsFeature(a.x + 1, a.y + 1, a.z)) {
      auto normal = _computeNormal(hermitianGrid.getValueAt(a).vertex_pos.scale(scale),
                                   hermitianGrid.getValueAt(b).vertex_pos.scale(scale),
                                   hermitianGrid.getValueAt(c).vertex_pos.scale(scale));
      normal = normal.scale(1 / normal.norm());
      _addVertex(normal, _normals);
      _addVertex(normal, _normals);
      _addVertex(normal, _normals);
      _addVertex(normal, _normals);
      _addVertex(hermitianGrid.getValueAt(a).vertex_pos.scale(scale), _vertices);
      _addVertex(hermitianGrid.getValueAt(b).vertex_pos.scale(scale), _vertices);
      _addVertex(hermitianGrid.getValueAt(c).vertex_pos.scale(scale), _vertices);
      _addVertex(hermitianGrid.getValueAt(d).vertex_pos.scale(scale), _vertices);
      _indices.push_back(vbo_idx);
      _indices.push_back(vbo_idx + 1);
      _indices.push_back(vbo_idx + 2);
      _indices.push_back(vbo_idx);
      _indices.push_back(vbo_idx + 2);
      _indices.push_back(vbo_idx + 3);
      vbo_idx += 4;
    }
  }
}


