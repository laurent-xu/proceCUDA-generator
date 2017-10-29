//
// Created by leo on 10/9/17.
//

#pragma once

#include <vector>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <rendering/node.hpp>

namespace rendering {

  namespace utils {

    class nmMatrix {
      public:
        template <typename T>
        static std::vector<T> multiply(const std::vector<T> &A, const std::vector<T> &B,
                                         int n1, int m1, int n2, int m2);

        template <typename T>
        static std::vector<T> transpose(const std::vector<T> &A, int n, int m);

        template <typename T>
        static std::vector<T> add(const std::vector<T> &A, const std::vector<T> &B, int n, int m);

        template <typename T>
        static std::vector<T> append(const std::vector<T> &A, const std::vector<T> &B, int n, int m1, int m2);

        template <typename T>
        static std::vector<T> extract(const std::vector<T> &A, int x, int y, int x_max, int y_max, int m);

        template <typename T>
        static void print(const std::vector<T> &A, int n, int m);

        template <typename T>
        static void print(const std::vector<T> &A, int n, int m, int space);
    };

  }

}

#include "nm-matrix.hxx"
