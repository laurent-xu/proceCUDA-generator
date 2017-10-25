//
// Created by leo on 10/12/17.
//

#include <tgmath.h>
#include <rendering/utils/nm-matrix.hpp>
#include "qr-decomposition.hpp"

namespace rendering {

  QRDecomposition::QRDecomposition(std::vector<data_t> matrix, int n, int m) : _matrix(matrix), _n(n), _m(m) {
    data_t aij = 0, ajj = 0, r = 0, s = 0, c = 0, ajk = 0, aik = 0;
    for (int j = 0; j < _m - 1; j++) {
      for (int i = j + 1; i < _n; i++) {
        aij = _matrix[i * _m + j];
        if (aij == 0)
          continue;
        ajj = _matrix[j * _m + j];
        r = std::hypot(ajj, aij);
        if (aij < 0)
          r *= -1;
        s = aij / r;
        c = ajj / r;
        for (int k = j; k < _m; k++) {
          ajk = _matrix[j * _m + k];
          aik = _matrix[i * _m + k];
          _matrix[j * _m + k] = c * ajk + s * aik;
          _matrix[i * _m + k] = -s * ajk + c * aik;
        }
      }
    }
  }

  std::vector<data_t> QRDecomposition::getProcessedMatrix() {
    return _matrix;
  }

  std::vector<data_t> QRDecomposition::extractAa() {
    return utils::nmMatrix<data_t>::extract(_matrix, 0, 0, _m - 1, std::min(_m - 1, _n), _m);
  }

  std::vector<data_t> QRDecomposition::extractBb() {
    return utils::nmMatrix<data_t>::extract(_matrix, _m - 1, 0, _m, _m - 1, _m);
  }

  data_t QRDecomposition::getR() {
    return _matrix[(_m - 1) * _m + _m - 1];
  }

}
