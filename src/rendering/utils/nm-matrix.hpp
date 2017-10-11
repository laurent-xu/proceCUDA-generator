//
// Created by leo on 10/9/17.
//

#pragma once

#include <vector>

namespace rendering {

  namespace utils {

    class nmMatrix {
      public:
        static std::vector<int> multiply(const std::vector<int> &A, const std::vector<int> &B,
                                         int n1, int m1, int n2, int m2);
        static std::vector<int> transpose(const std::vector<int> &A, int n, int m);
        static std::vector<int> add(const std::vector<int> &A, const std::vector<int> &B, int n, int m);
        static void print(const std::vector<int> &A, int n, int m);
    };

  }

}
