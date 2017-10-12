//
// Created by leo on 10/12/17.
//

#include <tgmath.h>
#include "qr-decomposition.hpp"

QRDecomposition::QRDecomposition(std::vector<double> matrix) : _matrix(matrix) {
  double aij = 0, ajj = 0, r = 0, s = 0, c = 0, ajk = 0, aik = 0;
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

std::vector<double> QRDecomposition::getR() {
  std::vector<double> result = _matrix;
  return result;
}

std::vector<double> QRDecomposition::getQ() {
  std::vector<double> result = _matrix;
  return result;
}
