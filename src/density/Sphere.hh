#pragma once
#include "F3Grid.hh"
#include "../utils/glm.hh"

static inline GridF3::grid_t make_sphere_example(const F3::vec3_t& grid_origin,
                                                 const F3::dist_t precision,
                                                 const F3::vec3_t& center,
                                                 const F3::dist_t radius)
{
  size_t dimension = 32;
  auto result = GridF3::get_grid(precision, grid_origin, dimension);
  for (size_t x = 0; x < dimension; ++x)
    for (size_t y = 0; y < dimension; ++y)
      for (size_t z = 0; z < dimension; ++z)
      {
        auto position = result->to_position(x, y, z);
        auto& f3 = result->at(x, y, z);
        f3.val = glm::distance(center, position) - radius;
        f3.grad = glm::normalize_safe(position - center);
      }
  return result;
}

static inline GridF3 debug_example(const F3::vec3_t& grid_origin,
                                         const F3::dist_t precision,
                                         const F3::vec3_t& center)
{
  auto result = GridF3(precision, grid_origin);
  auto& f3 = result.at(16, 16, 16);
  f3.val = -0.5;
  f3.grad = glm::normalize_safe(center);
  /*
  auto& f4 = result.at(17, 16, 16);
  f4.val = -0.5;
  f4.grad = glm::normalize_safe(center);
  auto& f5 = result.at(18, 16, 16);
  f5.val = -0.5;
  f5.grad = glm::normalize_safe(center);
   */
  return result;
}
