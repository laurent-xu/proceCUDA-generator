#include <density/F3Grid.hh>
#include <utils/glm.hh>
#include "generation_kernel.hh"

F3 kernel_f3(const F3::vec3_t& position)
{
  auto center = F3::vec3_t(10., 10., 10.);
  double radius = 10.;
  F3 f3;
  f3.val = glm::distance(center, position) - radius;
  f3.grad = glm::normalize_safe(position - center);
  return f3;
}
