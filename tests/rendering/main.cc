#include <iostream>
#include <vector>
#include <rendering/dual-contouring.hh>
#include <rendering/utils/nm-matrix.hpp>
#include <rendering/qr-decomposition.hpp>
#include <density/Sphere.hh>

void testSphere() {
  auto sphere = make_sphere_example(F3::vec3_t(0, 0, 0), F3::dist_t(1), F3::vec3_t(5, 5, 0), F3::dist_t(5));
}

void testMatrixNM() {
  int n1 = 3, m1 = 4;
  int n2 = 4, m2 = 3;
  int a[n1 * m1] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  int b[n2 * m2] = { 4, 7, 3, 6, 9, 1, 11, 12, 10, 2, 8, 5 };
  std::vector<float> A;
  std::vector<float> B;
  A.assign(a, a + n1 * m1);
  B.assign(b, b + n2 * m2);
  std::cout << "Matrix A:" << std::endl;
  rendering::utils::nmMatrix<float>::print(A, n1, m1);
  std::cout << "Matrix B:" << std::endl;
  rendering::utils::nmMatrix<float>::print(B, n2, m2);
  std::cout << "Matrix C = A * B:" << std::endl;
  auto C = rendering::utils::nmMatrix<float>::multiply(A, B, n1, m1, n2, m2);
  rendering::utils::nmMatrix<float>::print(C, n1, m2);
  std::cout << "Matrix A^t:" << std::endl;
  auto At = rendering::utils::nmMatrix<float>::transpose(A, n1, m1);
  rendering::utils::nmMatrix<float>::print(At, m1, n1);
  std::cout << "Matrix D: A^t + B" << std::endl;
  auto D = rendering::utils::nmMatrix<float>::add(At, B, n2, m2);
  rendering::utils::nmMatrix<float>::print(D, n2, m2);
  std::cout << "Matrix E: D::B" << std::endl;
  auto E = rendering::utils::nmMatrix<float>::append(D, B, n2, m2, m2);
  rendering::utils::nmMatrix<float>::print(E, n2, m2 + m2);
  std::cout << "Matrix F: extract E" << std::endl;
  auto F = rendering::utils::nmMatrix<float>::extract(E, 2, 1, 4, 4, 6);
  rendering::utils::nmMatrix<float>::print(F, 2, 2);
}

void testQRDecomposition() {
  float a1[] = { 6, 5, 0, 5, 1, 4, 0, 4, 3 };
  std::vector<rendering::data_t> m1;
  m1.assign(a1, a1 + 9);
  std::cout << "Start matrix:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(m1, 3, 3, 12);
  std::cout << std::endl;
  rendering::QRDecomposition qrd1(m1, 3, 3);
  std::cout << "Processed matrix:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(qrd1.getProcessedMatrix(), 3, 3, 12);
  std::cout << std::endl;
  std::cout << "A^:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(qrd1.extractAa(), 2, 2, 12);
  std::cout << std::endl;
  std::cout << "B^:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(qrd1.extractBb(), 2, 1, 12);
  std::cout << std::endl;
  std::cout << "r: " << qrd1.getR() << std::endl;

  float a2[] = { -1, 9, 2, 8, 7, 5, 6, -5, 7, 2, -9, 0, 1, 2, -3 };
  std::vector<rendering::data_t> m2;
  m2.assign(a2, a2 + 15);
  std::cout << "Start matrix:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(m2, 3, 5, 12);
  std::cout << std::endl;
  rendering::QRDecomposition qrd2(m2, 3, 5);
  std::cout << "Processed matrix:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(qrd2.getProcessedMatrix(), 3, 5, 12);
  std::cout << std::endl;
  std::cout << "A^:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(qrd2.extractAa(), 3, 4, 12);
  std::cout << std::endl;
  std::cout << "b^:" << std::endl;
  rendering::utils::nmMatrix<rendering::data_t>::print(qrd2.extractBb(), 3, 1, 12);
  std::cout << std::endl;
}

void testHermiteanComputation() {
  std::vector<std::vector<rendering::node_t>> nodes(2);
  for (int z = 0; z < 2; z++)
    for (int y = 0; y < 5; y++)
      for (int x = 0; x < 5; x++) {
        int r = rand() % 3;
        if (r == 0)
          nodes[z].push_back(rendering::node_t(-1, rendering::point_t(0, 0, 0)));
        else
          nodes[z].push_back(rendering::node_t(+1, rendering::point_t(0, 0, 0)));
      }
  rendering::HermitianGrid::printDensityGrid(nodes, rendering::point_t(5, 5, 2));
  rendering::HermitianGrid g(nodes, rendering::point_t(5, 5, 2), 1);
  for (int z = 0; z < 2; z++)
    for (int y = 0; y < 5; y++) {
      for (int x = 0; x < 5; x++) {
        auto e = nodes[z][y * 5 + x].value;
        if (e == -1)
          std::cout << ". ";
        if (e == 0)
          std::cout << "O ";
        if (e == 1)
          std::cout << "0 ";
      }
      std::cout << std::endl;
    }
  std::cout << std::endl;
  for (int z = 0; z < 2; z++)
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
  //testMatrixNM();
  //testQRDecomposition();
  //testHermiteanComputation();
  testSphere();
  return 0;
}