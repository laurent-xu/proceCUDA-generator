//
// Created by leo on 10/9/17.
//

#include <vector>
#include <iosfwd>
#include <iostream>
#include "nm-matrix.hpp"

namespace rendering {

  namespace utils {

    template <typename T>
    std::vector<T> nmMatrix<T>::multiply(const std::vector<T> &A, const std::vector<T> &B,
                                        int n1, int m1, int n2, int m2)
    {
      std::vector<T> result((unsigned long) (n1 * m2));
      for (int i = 0; i < n1; i++) {
        for (int j = 0; j < m2; j++) {
          T val = 0;
          for (int k = 0; k < n2; k++)
            val += A[i * m1 + k] * B[k * m2 + j];
          result[i * m2 + j] = val;
        }
      }
      return result;
    }

    template <typename T>
    std::vector<T> nmMatrix<T>::transpose(const std::vector<T> &A, int n, int m) {
      std::vector<T> result((unsigned long) (n * m));
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          result[i * n + j] = A[j * m + i];
        }
      }
      return result;
    }

    template <typename T>
    std::vector<T> nmMatrix<T>::add(const std::vector<T> &A, const std::vector<T> &B, int n, int m) {
      std::vector<T> result((unsigned long) (n * m));
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          result[i * m + j] = A[i * m + j] + B[i * m + j];
        }
      }
      return result;
    }

    template <typename T>
    void nmMatrix<T>::print(const std::vector<T> &A, int n, int m) {
      print(A, n, m, 5);
    }

    template <typename T>
    void nmMatrix<T>::print(const std::vector<T> &A, int n, int m, int space) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
          std::cout << std::right << std::setw(space) << A[i * m + j];
        std::cout << std::endl;
      }
    }

    template <typename T>
    std::vector<T> nmMatrix<T>::append(const std::vector<T> &A, const std::vector<T> &B, int n, int m1, int m2) {
      std::vector<T> result;
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m1; i++)
          result.push_back(A[j * m1 + i]);
        for (int i = 0; i < m2; i++)
          result.push_back(B[j * m2 + i]);
      }
      return result;
    }

    template <typename T>
    std::vector<T> nmMatrix<T>::extract(const std::vector<T> &A, int x, int y, int x_max, int y_max, int m) {
      int n1 = y_max - y;
      int m1 = x_max - x;
      std::vector<T> result(n1 * m1);
      int i = 0, j = 0;
      for (int ity = y; ity < y_max; ity++) {
        for (int itx = x; itx < x_max; itx++) {
          result[i * m1 + j] = A[ity * m + itx];
          j++;
        }
        j = 0;
        i++;
      }
      return result;
    }

  }

}
