#pragma once
#include "F3Grid.hh"
#include "../utils/glm.hh"

static inline GridF3 make_sphere_example(const F3::vec3_t& grid_origin,
                                         const F3::dist_t precision,
                                         const F3::vec3_t& center,
                                         const F3::dist_t radius)
{
  auto result = GridF3(precision, grid_origin);
  for (size_t x = 0; x < result.dim_size(); ++x)
    for (size_t y = 0; x < result.dim_size(); ++x)
      for (size_t z = 0; x < result.dim_size(); ++x)
      {
        auto position = result.to_position(x, y, z);
        auto& f3 = result.at(x, y, z);
        f3.val = glm::distance(center, position) - radius;
        f3.grad = glm::normalize_safe(position - center);
      }
  return result;
}
