//
// Created by leo on 10/9/17.
//

#include <iostream>
#include <iomanip>
#include "dual-contouring.hh"

namespace rendering {

  HermitianGrid::HermitianGrid(const std::vector<std::vector<node_t>> &grid, point_t dimensions, int nodeSize)
      : _grid(grid), _dimensions(dimensions), _nodeSize(nodeSize)
  {
    _initSurfaceNodes();
    _computeIntersections();
  }

  void HermitianGrid::_initSurfaceNodes() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          node.size = _nodeSize;
          node.min = point_t(x * node.size, y * node.size, z * node.size);
          if (pointContainsFeature(x, y, z))
            node.value = 0;
        }
  }

  void HermitianGrid::_computeIntersections() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          if (node.value == 1)  // Interior node
            continue;
          if (node.value == 0) {  // Surface node
            if (x + 1 < _dimensions.x && getValueAt(x + 1, y, z).value == -1)
              node.intersections.x = node.min.x + node.size / 2;
            if (y + 1 < _dimensions.y && getValueAt(x, y + 1, z).value == -1)
              node.intersections.y = node.min.y + node.size / 2;
            if (z + 1 < _dimensions.z && getValueAt(x, y, z + 1).value == -1)
              node.intersections.z = node.min.z + node.size / 2;
          }
          else {  // Air node
            if (x + 1 < _dimensions.x && getValueAt(x + 1, y, z).value != -1)
              node.intersections.x = node.min.x + node.size / 2;
            if (y + 1 < _dimensions.y && getValueAt(x, y + 1, z).value != -1)
              node.intersections.y = node.min.y + node.size / 2;
            if (z + 1 < _dimensions.z && getValueAt(x, y, z + 1).value != -1)
              node.intersections.z = node.min.z + node.size / 2;
          }
        }
  }

  void HermitianGrid::_computeVertices() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {

        }
  }

  point_t HermitianGrid::_computeVerticeForNode(int x, int y, int z) {
    auto &node = _grid[z][y * _dimensions.x + x];
    return point_t();
  }

  bool HermitianGrid::pointContainsFeature(int x, int y, int z) {
    int nb_air = 0;
    int nb_solid = 0;
    for (int i = 0; i <= 1 && x + i < _dimensions.x; i += 1)
      for (int j = 0; j <= 1 && y + j < _dimensions.y; j += 1)
        for (int k = 0; k <= 1 && z + k < _dimensions.z; k += 1) {
          auto &node = getValueAt(x + i, y + j, z + k);
          if (node.value == -1) {
            if (nb_solid != 0)
              return true;
            else
              nb_air++;
          }
          if (node.value == 1) {
            if (nb_air != 0)
              return true;
            else
              nb_solid++;
          }
        }
    return false;
  }

  void HermitianGrid::printDensityGrid(const std::vector<std::vector<node_t>> &density_grid, point_t dimensions) {
    for (int z = 0; z < dimensions.z; z++) {
      std::cout << "-- z = " << z << std::endl;
      for (int y = 0; y < dimensions.y; y++) {
        for (int x = 0; x < dimensions.x; x++) {
          std::cout << std::right << std::setw(10)
                    << "[" << density_grid[z][y * dimensions.x + x].value << ", "
                    << "(" << density_grid[z][y * dimensions.x + x].gradient.x << ", "
                    << density_grid[z][y * dimensions.x + x].gradient.y << ", "
                    << density_grid[z][y * dimensions.x + x].gradient.z << ")]";
        }
        std::cout << std::endl;
      }
    }
  }

  void HermitianGrid::printHermitianGrid(const std::vector<std::vector<node_t>> &density_grid, point_t dimensions) {
    for (int z = 0; z < dimensions.z; z++) {
      std::cout << "-- z = " << z << std::endl;
      for (int y = 0; y < dimensions.y; y++) {
        for (int x = 0; x < dimensions.x; x++) {
          std::cout << std::right << std::setw(10)
                    << "[" << density_grid[z][y * dimensions.x + x].value << ", "
                    << "(" << density_grid[z][y * dimensions.x + x].intersections.x << ", "
                    << density_grid[z][y * dimensions.x + x].intersections.y << ", "
                    << density_grid[z][y * dimensions.x + x].intersections.z << ")]";
        }
        std::cout << std::endl;
      }
    }
  }

}
