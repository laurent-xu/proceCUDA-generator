//
// Created by leo on 10/9/17.
//

#pragma once

#include <vector>
#include <glm/detail/type_mat.hpp>

namespace rendering {
  struct point_t {
    point_t() {}
    point_t(float x, float y, float z) : x(x), y(y), z(z) {}
    float x = 0;
    float y = 0;
    float z = 0;
  };

  struct node_t {
    node_t(int value, point_t gradient) : value(value), gradient(gradient) {}
    int value = -1;
    point_t gradient;
    point_t min;
    point_t vertex_pos;
    point_t intersections;
    int vbo_idx;
  };

  class HermitianGrid {
    public:
      HermitianGrid(const std::vector<std::vector<node_t>> &grid, point_t dimensions, float nodeSize);

    private:
      void _initSurfaceNodes();
      void _computeIntersections();
      void _computeVertices();
      point_t _computeVerticeForNode(int x, int y, int z);
      void _registerIntersectionsForVertex(std::vector<float> &A, std::vector<float> &b,
                                           const std::vector<float> &N, const node_t &node,
                                           bool check_x, bool check_y, bool check_z);
      void _registerIntersectionsForAxis(std::vector<float> &A, std::vector<float> &b,
                                         const std::vector<float> &N, const node_t &node, int axis);

    public:
      bool pointContainsFeature(int x, int y, int z);
      const node_t &getValueAt(int x, int y, int z) { return _grid[z][y * _dimensions.x + x]; }
      point_t getDimensions() { return _dimensions; }

    public:
      static void printDensityGrid(const std::vector<std::vector<node_t>> &density_grid, point_t dimensions);
      static void printHermitianGrid(const std::vector<std::vector<node_t>> &density_grid, point_t dimensions);

    public:
      const std::vector<std::vector<node_t>> &getGrid() const { return _grid; }

    private:
      std::vector<std::vector<node_t>> _grid;  // _grid[z][y * _width + x]
      const std::vector<std::vector<node_t>> _densityGrid;  // _grid[z][y * _width + x]
      point_t _dimensions;
      float _nodeSize;
  };

  class DualContouring {

  };

}

