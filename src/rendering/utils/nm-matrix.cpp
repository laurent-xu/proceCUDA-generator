//
// Created by leo on 10/9/17.
//

#include <vector>
#include <iosfwd>
#include <iostream>
#include <iomanip>
#include "nm-matrix.hpp"

namespace rendering {

  namespace utils {

    std::vector<int> nmMatrix::multiply(const std::vector<int> &A, const std::vector<int> &B,
                                        int n1, int m1, int n2, int m2)
    {
      std::vector<int> result((unsigned long) (n1 * m2));
      for (int i = 0; i < n1; i++) {
        for (int j = 0; j < m2; j++) {
          int val = 0;
          for (int k = 0; k < n2; k++)
            val += A[i * m1 + k] * B[k * m2 + j];
          result[i * m2 + j] = val;
        }
      }
      return result;
    }

    std::vector<int> nmMatrix::transpose(const std::vector<int> &A, int n, int m) {
      std::vector<int> result((unsigned long) (n * m));
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          result[i * n + j] = A[j * m + i];
        }
      }
      return result;
    }

    void nmMatrix::print(const std::vector<int> &A, int n, int m) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
          std::cout << std::right << std::setw(5) << A[i * m + j];
        std::cout << std::endl;
      }
    }

    std::vector<int> nmMatrix::add(const std::vector<int> &A, const std::vector<int> &B, int n, int m) {
      std::vector<int> result((unsigned long) (n * m));
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          result[i * m + j] = A[i * m + j] + B[i * m + j];
        }
      }
      return result;
    }

  }

}
