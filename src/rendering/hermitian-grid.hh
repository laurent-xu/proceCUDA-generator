//
// Created by leo on 10/9/17.
//

#pragma once

#include <vector>
#include <glm/detail/type_mat.hpp>
#include <density/F3Grid.hh>
#include <rendering/node.hpp>
#include <GL/gl.h>
#include "options.hpp"

namespace rendering {

  class HermitianGrid {

    public:
      HermitianGrid(const std::vector<std::vector<node_t>> &grid, point_t dimensions, float nodeSize);
      HermitianGrid(const GridF3<false>::grid_t& gridF3, point_t dimensions, float nodeSize);

    private:
      void _initSurfaceNodes(const GridF3<false>::grid_t &gridF3);
      void _computeIntersections();
      data_t _computeIntersectionOffset(data_t a, data_t b);
      void _computeContouringVertices();
      void computeVertexInfo(int x, int y, int z);
      point_t _computeVerticeForNode(int x, int y, int z);
      void _registerIntersectionsForVertex(std::vector<data_t> &A, std::vector<data_t> &b,
                                           const std::vector<data_t> &N, const node_t &node,
                                           bool check_x, bool check_y, bool check_z);
      void _registerIntersectionsForAxis(std::vector<data_t> &A, std::vector<data_t> &b,
                                         const std::vector<data_t> &N, const node_t &node, int axis);

    public:
      bool pointContainsFeature(int x, int y, int z) const;
      bool isSurface(int x, int y, int z);
      const node_t &getValueAt(int x, int y, int z) const { return _grid[z][y * _dimensions.x + x]; }
      const node_t &getValueAt(point_t p) const { return _grid[p.z][p.y * _dimensions.x + p.x]; }
      point_t getDimensions() const { return _dimensions; }

    public:
      static void printDensityGrid(const std::vector<std::vector<node_t>> &density_grid, point_t dimensions);
      static void printHermitianGrid(const std::vector<std::vector<node_t>> &density_grid, point_t dimensions);

    private:

    public:
      const std::vector<std::vector<node_t>> &getGrid() const { return _grid; }

    private:
      std::vector<std::vector<node_t>> _grid;  // _grid[z][y * _width + x]
      std::vector<std::vector<node_t>> _densityGrid;  // _grid[z][y * _width + x]
      point_t _dimensions;
      float _nodeSize;
  };

}

