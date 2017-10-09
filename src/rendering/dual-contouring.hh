//
// Created by leo on 10/9/17.
//

#pragma once

#include <vector>
#include <glm/detail/type_mat.hpp>

namespace rendering {
  struct point_t {
    point_t() {}
    point_t(int x, int y, int z) : x(x), y(y), z(z) {}
    int x = 0;
    int y = 0;
    int z = 0;
  };

  struct node_t {
    node_t(int value, point_t gradient) : value(value), gradient(gradient) {}
    int value = -1;
    point_t gradient;
    point_t min;
    point_t vertex_pos;
    point_t intersections;
    int size = 1;
    int vbo_idx;
  };

  class HermitianGrid {
    public:
      HermitianGrid(std::vector<std::vector<node_t>> grid, point_t dimensions, int nodeSize);

    private:
      void _initSurfaceNodes();
      void _computeIntersections();
      void _computeVertices();
      point_t _computeVerticeForNode(int x, int y, int z);

    public:
      node_t getValueAt(int x, int y, int z) { return _grid[z][y * _dimensions.x + x]; }
      point_t getDimensions() { return _dimensions; }

    private:
      std::vector<std::vector<node_t>> _grid;  // _grid[z][y * _width + x]
      point_t _dimensions;
      int _nodeSize = 1;
  };

  class DualContouring {

  };

}

