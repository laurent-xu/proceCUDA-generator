//
// Created by leo on 10/9/17.
//

#include <iostream>
#include <iomanip>
#include <rendering/utils/nm-matrix.hpp>
#include <tgmath.h>
#include "hermitian-grid.hh"
#include "qr-decomposition.hpp"

namespace rendering {

  HermitianGrid::HermitianGrid(const std::vector<std::vector<node_t>> &, point_t dimensions, float nodeSize)
      : _dimensions(dimensions), _nodeSize(nodeSize) {
    _initSurfaceNodes();
    computeVBOIndices();
    _computeIntersections();
    _computeContouringVertices(); // TODO: dual contouring
  }


  HermitianGrid::HermitianGrid(const GridF3<false>::grid_t& gridF3,
                               point_t dimensions, float nodeSize)
      : _dimensions(dimensions), _nodeSize(nodeSize)
  {
    for (size_t k = 0; k < dimensions.z; k++) {
      _densityGrid.push_back(std::vector<node_t>());
      for (size_t i = 0; i < dimensions.y; i++)
        for (size_t j = 0; j < dimensions.x; j++)
          _densityGrid[k].push_back(gridF3->at(j, i, k));
    }
    _grid = _densityGrid;
    _initSurfaceNodes();
    computeVBOIndices();
    _computeIntersections();
    _computeContouringVertices(); // TODO: dual contouring
  }

  void HermitianGrid::_initSurfaceNodes() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          node.min = point_t(x * _nodeSize, y * _nodeSize, z * _nodeSize);
        }
  }

  void HermitianGrid::_computeIntersections() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          auto &densityNode = _densityGrid[z][y * _dimensions.x + x];
          // TODO: Value = 0 ?
          if (x + 1 < _dimensions.x && _densityGrid[z][y * _dimensions.x + x + 1].value * densityNode.value < 0)
            node.intersections.x = node.min.x + _computeIntersectionOffset(
                densityNode.value, _densityGrid[z][y * _dimensions.x + x + 1].value);
          if (y + 1 < _dimensions.y && _densityGrid[z][(y + 1) * _dimensions.x + x].value * densityNode.value < 0)
            node.intersections.y = node.min.y + _computeIntersectionOffset(
                densityNode.value, _densityGrid[z][(y + 1) * _dimensions.x + x].value);
          if (z + 1 < _dimensions.z && _densityGrid[z + 1][y * _dimensions.x + x].value * densityNode.value < 0)
            node.intersections.y = node.min.z + _computeIntersectionOffset(
                densityNode.value, _densityGrid[z + 1][y * _dimensions.x + x].value);
        }
  }

  data_t HermitianGrid::_computeIntersectionOffset(data_t a, data_t b) {
    return -a / b - a;
  }

  void HermitianGrid::_computeContouringVertices() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {
          auto &node = _grid[z][y * _dimensions.x + x];
          node.vertex_pos = point_t(
              (float) (node.min.x - (float) (_dimensions.x) / 2.0f),
              (float) (node.min.y - (float) (_dimensions.y) / 2.0f),
              (float) (node.min.z - (float) (_dimensions.z) / 2.0f)
          );
          /*
          if (node.value == 0)
            _computeVerticeForNode(x, y, z); // TODO: dual contouring
          */
        }
  }

  point_t HermitianGrid::_computeVerticeForNode(int x, int y, int z) {
    auto &node = _grid[z][y * _dimensions.x + x];
    data_t n[] = {node.gradient.x, node.gradient.y, node.gradient.z};
    std::vector<data_t> N;
    N.assign(n, n + 3);
    std::vector<data_t> A;
    std::vector<data_t> b;
    for (int i = 0; i <= 1 && x + i < _dimensions.x; i += 1)
      for (int j = 0; j <= 1 && y + j < _dimensions.y; j += 1)
        for (int k = 0; k <= 1 && z + k < _dimensions.z; k += 1) {
          _registerIntersectionsForVertex(A, b, N, getValueAt(x + i, y + j, z + k),
                                          i != 1, j != 1, k != 1);
        }
    auto Ab = utils::nmMatrix::append(A, b, (int) b.size(), 3, 1);
    QRDecomposition qrd(Ab, (int) (Ab.size() / 4), 4);
    auto QAb = qrd.getProcessedMatrix();
    auto xA = qrd.extractAa();
    utils::nmMatrix::print(xA, 3, 3, 12);
    auto xb = qrd.extractBb();
    utils::nmMatrix::print(xb, 3, 1, 12);
    auto r = qrd.getR();
    std::cout << r << std::endl;
    return point_t(0, 0, 0);
  }


  void HermitianGrid::_registerIntersectionsForVertex(std::vector<data_t> &A, std::vector<data_t> &b,
                                                      const std::vector<data_t> &N, const node_t &node,
                                                      bool check_x, bool check_y, bool check_z) {
    if (check_x && node.intersections.x != 0)
      _registerIntersectionsForAxis(A, b, N, node, 0);
    if (check_y && node.intersections.y != 0)
      _registerIntersectionsForAxis(A, b, N, node, 1);
    if (check_z && node.intersections.z != 0)
      _registerIntersectionsForAxis(A, b, N, node, 2);
  }

  void HermitianGrid::_registerIntersectionsForAxis(std::vector<data_t> &A, std::vector<data_t> &b,
                                                    const std::vector<data_t> &N, const node_t &node, int axis) {
    A.insert(A.end(), N.begin(), N.end());
    data_t p[3] = {node.min.x, node.min.y, node.min.z};
    data_t intersections_compo[3] = {node.intersections.x, node.intersections.y, node.intersections.z};
    p[axis] += intersections_compo[axis];
    std::vector<data_t> pi;
    pi.assign(p, p + 3);
    std::vector<data_t> ni = N;
    data_t np = utils::nmMatrix::multiply(ni, pi, 1, 3, 3, 1)[0];
    b.push_back(np);
  }


  bool HermitianGrid::pointContainsFeature(int x, int y, int z) const {
    int nb_air = 0;
    int nb_solid = 0;
    for (int i = 0; i <= 1 && x + i < _dimensions.x; i += 1)
      for (int j = 0; j <= 1 && y + j < _dimensions.y; j += 1)
        for (int k = 0; k <= 1 && z + k < _dimensions.z; k += 1) {
          auto &node = _densityGrid[z + k][(y + j) * _dimensions.x + x + i];
          if (node.value >= 0) {
            if (nb_solid != 0)
              return true;
            else
              nb_air++;
          }
          if (node.value < 0) {
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

  void HermitianGrid::computeVBOIndices() {
    for (int z = 0; z < _dimensions.z; z++)
      for (int y = 0; y < _dimensions.y; y++)
        for (int x = 0; x < _dimensions.x; x++) {}
  }

  bool HermitianGrid::isSurface(int x, int y, int z) {
    auto &densityNode = _densityGrid[z][y * _dimensions.x + x];
    return  (x + 1 < _dimensions.x && _densityGrid[z][y * _dimensions.x + x + 1].value * densityNode.value < 0)
            || (y + 1 < _dimensions.y && _densityGrid[z][(y + 1) * _dimensions.x + x].value * densityNode.value < 0)
            || (z + 1 < _dimensions.z && _densityGrid[z + 1][y * _dimensions.x + x].value * densityNode.value < 0);
  }

}

