//
// Created by leo on 10/12/17.
//

#pragma once


#include <vector>

class QRDecomposition {
  public:
    QRDecomposition(std::vector<double> matrix);
    std::vector<double> getR();
    std::vector<double> getQ();

  private:
    std::vector<double> _matrix;

};


