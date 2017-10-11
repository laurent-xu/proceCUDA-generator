//
// Created by leo on 10/9/17.
//

#include <iostream>
#include <iomanip>
#include <rendering/utils/nm-matrix.hpp>
#include "dual-contouring.hh"

namespace rendering {

  HermitianGrid::HermitianGrid(const std::vector<std::vector<node_t>> &grid, point_t dimensions, float nodeSize)
      : _grid(grid), _densityGrid(grid), _dimensions(dimensions), _nodeSize(nodeSize)
  {
    _initSurfaceNodes();
    _computeIntersections();
    _computeVertices();
  }

  void HermitianGrid::_initSurfaceNodes() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          node.min = point_t(x * _nodeSize, y * _nodeSize, z * _nodeSize);
          if (pointContainsFeature(x, y, z))
            node.value = 0;
        }
  }

  void HermitianGrid::_computeIntersections() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          auto &densityNode = _densityGrid[z][y * _dimensions.x + x];
          if (node.value == 0) {  // Surface node
            if (x + 1 < _dimensions.x && _densityGrid[z][y * _dimensions.x + x + 1].value != densityNode.value)
              node.intersections.x = _nodeSize / 2;
            if (y + 1 < _dimensions.y && _densityGrid[z][(y + 1) * _dimensions.x + x].value != densityNode.value)
              node.intersections.y =  _nodeSize / 2;
            if (z + 1 < _dimensions.z && _densityGrid[z + 1][y * _dimensions.x + x].value != densityNode.value)
              node.intersections.z = _nodeSize / 2;
          }
        }
  }

  void HermitianGrid::_computeVertices() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          if (node.value == 0)
            _computeVerticeForNode(x, y, z);
        }
  }

  point_t HermitianGrid::_computeVerticeForNode(int x, int y, int z) {
    auto &node = _grid[z][y * _dimensions.x + x];
    float n[] = { node.gradient.x, node.gradient.y, node.gradient.z };
    std::vector<float> N;
    N.assign(n, n + 3);
    std::vector<float> A;
    std::vector<float> b;
    for (int i = 0; i <= 1 && x + i < _dimensions.x; i += 1)
      for (int j = 0; j <= 1 && y + j < _dimensions.y; j += 1)
        for (int k = 0; k <= 1 && z + k < _dimensions.z; k += 1) {
          _registerIntersectionsForVertex(A, b, N, getValueAt(x + i, y + j, z + k),
                                          i != 1, j != 1, k != 1);
        }
    /*
    std::cout << "RESULTS" << std::endl;
    std::cout << "\tA:" << std::endl;
    utils::nmMatrix<float>::print(A, (int) (A.size() / 3), 3);
    std::cout << "\tb:" << std::endl;
    utils::nmMatrix<float>::print(b, (int) b.size(), 1);
    */
    return point_t(0, 0, 0);
  }


  void HermitianGrid::_registerIntersectionsForVertex(std::vector<float> &A, std::vector<float> &b,
                                                      const std::vector<float> &N, const node_t &node,
                                                      bool check_x, bool check_y, bool check_z)
  {
    if (check_x && node.intersections.x != 0)
      _registerIntersectionsForAxis(A, b, N, node, 0);
    if (check_y && node.intersections.y != 0)
      _registerIntersectionsForAxis(A, b, N, node, 1);
    if (check_z && node.intersections.z != 0)
      _registerIntersectionsForAxis(A, b, N, node, 2);
  }

  void HermitianGrid::_registerIntersectionsForAxis(std::vector<float> &A, std::vector<float> &b,
                                                    const std::vector<float> &N, const node_t &node, int axis) {
    A.insert(A.end(), N.begin(), N.end());
    float p[3] = { node.min.x, node.min.y, node.min.z };
    float intersections_compo[3] = { node.intersections.x, node.intersections.y, node.intersections.z};
    p[axis] += intersections_compo[axis];
    std::vector<float> pi;
    pi.assign(p, p + 3);
    std::vector<float> ni = N;
    float np = utils::nmMatrix<float>::multiply(ni, pi, 1, 3, 3, 1)[0];
    b.push_back(np);
  }


  bool HermitianGrid::pointContainsFeature(int x, int y, int z) {
    int nb_air = 0;
    int nb_solid = 0;
    for (int i = 0; i <= 1 && x + i < _dimensions.x; i += 1)
      for (int j = 0; j <= 1 && y + j < _dimensions.y; j += 1)
        for (int k = 0; k <= 1 && z + k < _dimensions.z; k += 1) {
          auto &node = _densityGrid[z + k][(y + j) * _dimensions.x + x + i];
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
