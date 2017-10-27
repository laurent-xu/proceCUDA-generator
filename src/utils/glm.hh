#pragma once
#include <glm/glm.hpp>
#define EPSILON 1e-6

namespace glm
{
  template <typename T>
  T normalize_safe(const T& v)
  {
    auto len = glm::length(v);
    if (len < EPSILON)
      return glm::normalize(v);
    return T();
  }
}
