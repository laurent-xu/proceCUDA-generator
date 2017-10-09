//
// Created by leo on 10/9/17.
//

#include <iostream>
#include "dual-contouring.hh"

namespace rendering {

  HermitianGrid::HermitianGrid(std::vector<std::vector<node_t>> grid, point_t dimensions, int nodeSize)
      : _grid(grid), _dimensions(dimensions), _nodeSize(nodeSize)
  {
    _initSurfaceNodes();
    _computeIntersections();
  }

  void HermitianGrid::_initSurfaceNodes() {
    int SIZE = 1;
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          node.size = SIZE;
          node.min = point_t(x * node.size, y * node.size, z * node.size);
          if (node.value == -1)
            continue;
          if (x - 1 >= 0 && getValueAt(x - 1, y, z).value == -1
              || x + 1 < _dimensions.x && getValueAt(x + 1, y, z).value == -1
              || y - 1 >= 0 && getValueAt(x, y - 1, z).value == -1
              || y + 1 < _dimensions.y && getValueAt(x, y + 1, z).value == -1
              || z - 1 >= 0 && getValueAt(x, y, z - 1).value == -1
              || z + 1 < _dimensions.z && getValueAt(x, y, z + 1).value == -1)
          {
            node.value = 0;
          }
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
  }

  point_t HermitianGrid::_computeVerticeForNode(int x, int y, int z) {
    auto &node = _grid[z][y * _dimensions.x + x];
    return point_t();
  }

}
