#include <iostream>
#include <vector>
#include <rendering/dual-contouring.hh>
#include <rendering/utils/nm-matrix.hpp>

void testMatrixNM() {
  int n1 = 3, m1 = 4;
  int n2 = 4, m2 = 3;
  int a[n1 * m1] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  int b[n2 * m2] = { 4, 7, 3, 6, 9, 1, 11, 12, 10, 2, 8, 5 };
  std::vector<int> A;
  std::vector<int> B;
  A.assign(a, a + n1 * m1);
  B.assign(b, b + n2 * m2);
  std::cout << "Matrix A:" << std::endl;
  rendering::utils::nmMatrix::print(A, n1, m1);
  std::cout << "Matrix B:" << std::endl;
  rendering::utils::nmMatrix::print(B, n2, m2);
  std::cout << "Matrix C = A * B:" << std::endl;
  auto C = rendering::utils::nmMatrix::multiply(A, B, n1, m1, n2, m2);
  rendering::utils::nmMatrix::print(C, n1, m2);
  std::cout << "Matrix A^t:" << std::endl;
  auto At = rendering::utils::nmMatrix::transpose(A, n1, m1);
  rendering::utils::nmMatrix::print(At, m1, n1);
  std::cout << "Matrix D: A^t + B" << std::endl;
  auto D = rendering::utils::nmMatrix::add(At, B, n2, m2);
  rendering::utils::nmMatrix::print(D, n2, m2);
}

void testHermiteanComputation() {
  std::vector<std::vector<rendering::node_t>> nodes(1);
  for (int z = 0; z < 1; z++)
    for (int y = 0; y < 5; y++)
      for (int x = 0; x < 5; x++) {
        int r = rand() % 3;
        if (r == 0)
          nodes[z].push_back(rendering::node_t(-1, rendering::point_t(0, 0, 0)));
        else
          nodes[z].push_back(rendering::node_t(+1, rendering::point_t(0, 0, 0)));
      }
  rendering::HermitianGrid::printDensityGrid(nodes, rendering::point_t(5, 5, 1));
  rendering::HermitianGrid g(nodes, rendering::point_t(5, 5, 1), 1);
  for (int z = 0; z < 1; z++)
    for (int y = 0; y < 5; y++) {
      for (int x = 0; x < 5; x++) {
        auto e = g.getValueAt(x, y, z).value;
        if (e == -1)
          std::cout << ". ";
        if (e == 0)
          std::cout << "O ";
        if (e == 1)
          std::cout << "0 ";
      }
      std::cout << std::endl;
    }
  rendering::HermitianGrid::printHermitianGrid(g.getGrid(), g.getDimensions());
}

int main() {
  testHermiteanComputation();
  return 0;
}