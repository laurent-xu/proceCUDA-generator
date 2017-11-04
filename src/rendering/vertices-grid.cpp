//
// Created by leo on 11/4/17.
//

#include <iostream>
#include "vertices-grid.hpp"

namespace rendering {

  VerticesGrid::VerticesGrid(const HermitianGrid &hermitianGrid, float scale) {
    computeVBO(hermitianGrid, scale);
  }

  void VerticesGrid::computeNormals() {
    for (size_t i = 0; i < _vertices.size(); i += 9)
      _computeNormal(point_t(_vertices[i], _vertices[i + 1], _vertices[i + 2]),
                     point_t(_vertices[i + 3], _vertices[i + 4], _vertices[i + 5]),
                     point_t(_vertices[i + 6], _vertices[i + 7], _vertices[i + 8]));
  }

  void
  VerticesGrid::_computeNormal(const point_t &p1, const point_t &p2, const point_t &p3) {
    auto U = p2 - p1;
    auto V = p3 - p1;
    data_t x = U.y * V.z - U.z * V.y;
    data_t y = U.z * V.x - U.x * V.z;
    data_t z = U.x * V.y - U.y * V.x;
    _normals.push_back(x);
    _normals.push_back(y);
    _normals.push_back(z);

    _normals.push_back(x);
    _normals.push_back(y);
    _normals.push_back(z);

    _normals.push_back(x);
    _normals.push_back(y);
    _normals.push_back(z);
  }

  void VerticesGrid::computeVBO(const HermitianGrid &hermitianGrid, float scale) {
    for (int z = 0; z < hermitianGrid.getDimensions().z; z++)
      for (int y = 0; y < hermitianGrid.getDimensions().y; y++)
        for (int x = 0; x < hermitianGrid.getDimensions().x; x++)
          if (hermitianGrid.pointContainsFeature(x, y, z)) {
            auto &node = hermitianGrid.getValueAt(x, y, z);
            _addVertex(point_t(
                (float) ((node.min.x - (float) (hermitianGrid.getDimensions().x) / 2.0f) * scale),
                (float) ((node.min.y - (float) (hermitianGrid.getDimensions().y) / 2.0f) * scale),
                (float) ((node.min.z - (float) (hermitianGrid.getDimensions().z) / 2.0f) * scale)
            ), _vertices);
          }
    for (int z = 0; z < hermitianGrid.getDimensions().z; z++) {
      for (int y = 0; y < hermitianGrid.getDimensions().y; y++) {
        for (int x = 0; x < hermitianGrid.getDimensions().x; x++) {
          auto &node = hermitianGrid.getValueAt(x, y, z);
          if (node.vbo_idx != -1) {
            if (x + 1 < hermitianGrid.getDimensions().x && hermitianGrid.getValueAt(x + 1, y, z).vbo_idx != -1
                && y + 1 < hermitianGrid.getDimensions().y && hermitianGrid.getValueAt(x + 1, y + 1, z).vbo_idx != -1) {
              _indices.push_back(hermitianGrid.getValueAt(x, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x + 1, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x + 1, y + 1, z).vbo_idx);
            }
            if (y + 1 < hermitianGrid.getDimensions().y && hermitianGrid.getValueAt(x, y + 1, z).vbo_idx != -1
                && x + 1 < hermitianGrid.getDimensions().x && hermitianGrid.getValueAt(x + 1, y + 1, z).vbo_idx != -1) {
              _indices.push_back(hermitianGrid.getValueAt(x, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x + 1, y + 1, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x, y + 1, z).vbo_idx);
            }
            if (x + 1 < hermitianGrid.getDimensions().x && hermitianGrid.getValueAt(x + 1, y, z).vbo_idx != -1
                && z + 1 < hermitianGrid.getDimensions().z && hermitianGrid.getValueAt(x + 1, y, z + 1).vbo_idx != -1) {
              _indices.push_back(hermitianGrid.getValueAt(x, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x + 1, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x + 1, y, z + 1).vbo_idx);
            }
            if (z + 1 < hermitianGrid.getDimensions().z && hermitianGrid.getValueAt(x, y, z + 1).vbo_idx != -1
                && x + 1 < hermitianGrid.getDimensions().x && hermitianGrid.getValueAt(x + 1, y, z + 1).vbo_idx != -1) {
              _indices.push_back(hermitianGrid.getValueAt(x, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x + 1, y, z + 1).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x, y, z + 1).vbo_idx);
            }
            if (y + 1 < hermitianGrid.getDimensions().y && hermitianGrid.getValueAt(x, y + 1, z).vbo_idx != -1
                && z + 1 < hermitianGrid.getDimensions().z && hermitianGrid.getValueAt(x, y + 1, z + 1).vbo_idx != -1) {
              _indices.push_back(hermitianGrid.getValueAt(x, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x, y + 1, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x, y + 1, z + 1).vbo_idx);
            }
            if (z + 1 < hermitianGrid.getDimensions().z && hermitianGrid.getValueAt(x, y, z + 1).vbo_idx != -1
                && y + 1 < hermitianGrid.getDimensions().y && hermitianGrid.getValueAt(x, y + 1, z + 1).vbo_idx != -1) {
              _indices.push_back(hermitianGrid.getValueAt(x, y, z).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x, y + 1, z + 1).vbo_idx);
              _indices.push_back(hermitianGrid.getValueAt(x, y, z + 1).vbo_idx);
            }
          }
        }
      }
    }
    computeNormals();
    // std::cout << "normals " << _normals.size() << std::endl;
    // std::cout << "vertices " << _vertices.size() << std::endl;
    for (size_t i = 0; i < _vertices.size(); i += 3) {
      _vbo.push_back(_vertices[i]);
      _vbo.push_back(_vertices[i + 1]);
      _vbo.push_back(_vertices[i + 2]);
      _vbo.push_back(_normals[i]);
      _vbo.push_back(_normals[i + 1]);
      _vbo.push_back(_normals[i + 2]);
    }
  }

  void VerticesGrid::_addVertex(point_t vertex, std::vector<GLfloat> &buffer_vect) {
    buffer_vect.push_back(vertex.x);
    buffer_vect.push_back(vertex.y);
    buffer_vect.push_back(vertex.z);
  }
}

