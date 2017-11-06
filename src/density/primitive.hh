#pragma once
#include <density/F3Grid.hh>
#include <utils/cudamacro.hh>
#include <utils/glm.hh>

namespace density
{
  DEVICE_TARGET inline static F3 add(const F3& in)
  {
    return in;
  }

  template <typename... Args>
  DEVICE_TARGET inline static F3 add(const F3& in, Args&&... args)
  {
    return in + add(std::forward<Args>(args)...);
  }

  DEVICE_TARGET inline static F3 clamp(double min, double max, const F3& in)
  {
    F3 out = F3{0., F3::vec3_t(0., 0., 0.)};
    if (in.val < min)
      out.val = min;
    else if(in.val > max)
      out.val = max;
    else
      out = in;
    return out;
  }

  DEVICE_TARGET inline static F3 multiply(const F3& in)
  {
    return in;
  }

  template <typename... Args>
  DEVICE_TARGET inline static F3 multiply(const F3& in, Args&&... args)
  {
    return in * add(std::forward<Args>(args)...);
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

  DEVICE_TARGET inline static F3 polynom(const F3&, double c)
  {
    return F3{c, F3::vec3_t{0., 0., 0.}};
  }

  template <typename... Args>
  DEVICE_TARGET inline static F3 polynom(const F3& in, double c, Args&&... args)
  {
    return F3{c, F3::vec3_t{0., 0., 0.}} +
           in * polynom(in, std::forward<Args>(args)...);
  }

  DEVICE_TARGET
  inline static F3::vec3_t project_spatialize(const F3::vec3_t& center,
                                              double radius,
                                              const F3::vec3_t& position)
  {
    return glm::normalize_safe(position - center) * radius;
  }

  DEVICE_TARGET inline static F3 spatialize(const F3& height,
                                            double radius,
                                            const F3::vec3_t& center_to_pos)
  {
    F3 in;
    in.grad = height.grad * (height.val + radius) / radius;
    in.grad -= glm::dot(in.grad, glm::normalize_safe(center_to_pos));
    in.val = height.val - radius;
    return in;
  }

  DEVICE_TARGET
  inline static F3 sphere(const F3::vec3_t& center, double radius,
                          const F3::vec3_t& position)
  {
    return F3{glm::distance(position, center) - radius,
              glm::normalize_safe(position - center)};
  }
}
