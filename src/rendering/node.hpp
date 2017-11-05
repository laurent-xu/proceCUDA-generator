//
// Created by leo on 10/29/17.
//

#ifndef PROCECUDA_NODE_HPP
#define PROCECUDA_NODE_HPP

#include "options.hpp"
#include <density/F3Grid.hh>

namespace rendering {
  struct point_t {
    using vec3_t = F3::vec3_t;
    point_t() {}
    point_t(vec3_t vec) : x(vec.x), y(vec.y), z(vec.z) {}
    point_t(float a) : x(a), y(a), z(a) {}
    point_t(float x, float y, float z) : x(x), y(y), z(z) {}
    data_t x = 0;
    data_t y = 0;
    data_t z = 0;

    float norm() {
      return sqrt(x * x + y * y + z * z);
    }

    point_t scale(float scale) const {
      auto p = point_t(x, y, z);
      p.x *= scale;
      p.y *= scale;
      p.z *= scale;
      return p;
    }

    point_t &operator+=(const point_t &other) {
      this->x += other.x;
      this->y += other.y;
      this->z += other.z;
      return *this;
    }

    point_t &operator-=(const point_t &other) {
      this->x -= other.x;
      this->y -= other.y;
      this->z -= other.z;
      return *this;
    }

    const point_t operator+(const point_t &other) const {
      auto result = *this;
      result += other;
      return result;
    }

    const point_t operator-(const point_t &other) const {
      auto result = *this;
      result -= other;
      return result;
    }

  };

  inline std::ostream &operator<<(std::ostream &os, const point_t &p) {
    os << "(" << p.x << "," << p.y << "," << p.z << ")";
    return os;
  }

  struct node_t {
    node_t(int value, point_t gradient) : value(value), gradient(gradient) {}
    node_t(const F3 &f3) : value(f3.val), gradient(f3.grad) {}

    data_t value = -1;
    point_t gradient;
    point_t min;
    point_t vertex_pos;
    point_t intersections;
  };

  inline std::ostream &operator<<(std::ostream &os, const node_t &n) {
    os << "(" << n.value << "/" << n.gradient << "/" << n.min << "/" << n.intersections << ")";
    return os;
  }
}

#endif //PROCECUDA_NODE_HPP
