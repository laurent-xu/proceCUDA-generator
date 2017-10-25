//
// Created by leo on 10/12/17.
//

#pragma once


#include <vector>
#include "options.hpp"

namespace rendering {

    class QRDecomposition {

      public:
        QRDecomposition(std::vector<data_t> matrix, int n, int m);
        ~QRDecomposition() {}
        std::vector<data_t> getProcessedMatrix();
        std::vector<data_t> extractAa();
        std::vector<data_t> extractBb();
        data_t getR();

      private:
        std::vector<data_t> _matrix;
        int _n;
        int _m;

    };

}
