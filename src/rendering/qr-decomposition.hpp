//
// Created by leo on 10/12/17.
//

#pragma once


#include <vector>
#include "options.hpp"

namespace rendering {

    class QRDecomposition {

      public:
        QRDecomposition(std::vector<data_t> matrix);

        std::vector<data_t> getR();

        std::vector<data_t> getQ();

      private:
        std::vector<data_t> _matrix;

    };

}
