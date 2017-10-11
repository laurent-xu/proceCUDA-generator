//
// Created by leo on 10/9/17.
//

#pragma once

#include <vector>
#include <iomanip>

namespace rendering {

  namespace utils {

    template <typename T>
    class nmMatrix {
      public:
        static std::vector<T> multiply(const std::vector<T> &A, const std::vector<T> &B,
                                         int n1, int m1, int n2, int m2);
        static std::vector<T> transpose(const std::vector<T> &A, int n, int m);
        static std::vector<T> add(const std::vector<T> &A, const std::vector<T> &B, int n, int m);
        static std::vector<T> append(const std::vector<T> &A, const std::vector<T> &B, int n, int m1, int m2);
        static void print(const std::vector<T> &A, int n, int m);
    };

  }

}

#include "nm-matrix.hxx"
