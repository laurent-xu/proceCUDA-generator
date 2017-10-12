//
// Created by leo on 10/12/17.
//

#include <tgmath.h>
#include <math_functions.h>
#include "qr-decomposition.hpp"

namespace rendering {

  QRDecomposition::QRDecomposition(std::vector<data_t> matrix) : _matrix(matrix) {
    data_t aij = 0, ajj = 0, r = 0, s = 0, c = 0, ajk = 0, aik = 0;
    for (int j = 0; j < 2; j++) {
      for (int i = j + 1; i < _matrix.size() / 3; i++) {
        aij = _matrix[i * 3 + j];
        if (aij == 0)
          continue;
        ajj = _matrix[j * 3 + j];
        r = std::hypot(ajj, aij);
        if (aij < 0)
          r *= -1;
        s = aij / r;
        c = ajj / r;
        for (int k = j; k < 3; k++) {
          ajk = _matrix[j * 3 + k];
          aik = _matrix[i * 3 + k];
          _matrix[j * 3 + k] = c * ajk + s * aik;
          _matrix[i * 3 + k] = -s * ajk + c * aik;
        }
      }
    }
  }

  std::vector<data_t> QRDecomposition::getR() {
    std::vector<data_t> result = _matrix;
    return result;
  }

  std::vector<data_t> QRDecomposition::getQ() {
    int m = std::max((int) (_matrix.size() / 3), 3);
    std::vector<data_t> Q((unsigned long) (m * m));
    for (int i = 0; i < m; i++)
      Q[i * m + i] = 1;

    data_t aij, c, s, jk, ik = 0;
    for (int j = 2; j >= 0; j--) {
      for (int i = (int) (_matrix.size() / 3) - 1; i > j; i--) {
        // Get c and s which are stored in the i-th row, j-th column.
        aij = _matrix[i * 3 + j];
        data_t abs_aij = aij > 0 ? aij : -aij;
        if (aij == 0) {
          c = 0;
          s = 1;
        } else if (abs_aij < 1) {
          s = 2. * abs_aij;
          c = sqrt(1 - pow(s, 2));
          if (aij < 0) {
            c = -c;
          }
        } else {
          c = 2 / aij;
          s = sqrt(1 - pow(c, 2));
        }

        for (int k = 0; k < 3; k++) {
          jk = Q[j * 3 + k];
          ik = Q[i * 3 + k];
          Q[j * 3 + k] = c * jk - s * ik;
          Q[i * 3 + k] = s * jk + c * ik;
        }
      }
    }
    return Q;
  }

}
