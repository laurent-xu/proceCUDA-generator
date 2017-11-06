#pragma once
#include <density/F3Grid.hh>
#include <utils/cudamacro.hh>

namespace density
{
  DEVICE_TARGET inline static F3 add(const F3&)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  template <typename... Args>
  DEVICE_TARGET inline static F3 add(const F3&, Args&&...)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  DEVICE_TARGET inline static F3 clamp(double, double, const F3&)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  DEVICE_TARGET inline static F3 multiply(const F3&)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  template <typename... Args>
  DEVICE_TARGET inline static F3 multiply(const F3&, Args&&...)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  DEVICE_TARGET
  inline static F3 perlin(const F3::vec3_t& position,
                          double normalization, size_t seed)
  {
    (void) position;
    (void) normalization;
    (void) seed;
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  DEVICE_TARGET inline static F3 polynom(const F3&, double)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  template <typename... Args>
  DEVICE_TARGET inline static F3 polynom(const F3&, Args&&...)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  DEVICE_TARGET
  inline static F3::vec3_t project_spatialize(const F3::vec3_t&, double,
                                              const F3::vec3_t&)
  {
    return F3::vec3_t{0., 0., 0.};
  }

  DEVICE_TARGET inline static F3 spatialize(const F3&, double)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }

  DEVICE_TARGET
  inline static F3 sphere(const F3::vec3_t&, double, const F3::vec3_t&)
  {
    return F3{0, F3::vec3_t{0., 0., 0.}};
  }
}
