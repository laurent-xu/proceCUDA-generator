#include <density/F3Grid.hh>
#include <utils/glm.hh>
#include "generation_kernel.hh"

__global__ void kernel_f3(GridF3<true> grid)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
  size_t dimension = grid.dim_size();

  if (x < dimension && y < dimension && z < dimension)
  {
    auto center = F3::vec3_t(16., 16., 32.);
    double radius = 10.;

    auto position = grid.to_position(x, y, z);
    F3& f3 = grid.at(x, y, z);
    f3.val = glm::distance(center, position) - radius;
    f3.grad = glm::normalize_safe(position - center);
  }
}
