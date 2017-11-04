#pragma once
#include <density/F3Grid.hh>
#include <vector>

#ifdef CUDA_GENERATION
__global__ void kernel_f3(GridF3<true> grid);
GridF3<true>::grid_t make_density_grid_aux(const GridInfo& info)
{
  size_t dimension = info.dimension;
  size_t thread_per_dim = 8;
  size_t block_dim = (dimension + thread_per_dim - 1) / thread_per_dim;
  dim3 Dg(block_dim, block_dim, block_dim);
  dim3 Db(thread_per_dim, thread_per_dim, thread_per_dim);

  auto result = GridF3<true>::get_grid(info);
  kernel_f3<<<Dg,Db>>>(*result);
  // TODO Remove the next line
  cudaDeviceSynchronize();
  return result;
}
#else
F3 kernel_f3(const F3::vec3_t& position);
GridF3<false>::grid_t make_density_grid_aux(const GridInfo& info)
{
  size_t dimension = info.dimension;
  auto result = GridF3<false>::get_grid(info);
  for (size_t x = 0; x < dimension; ++x)
    for (size_t y = 0; y < dimension; ++y)
      for (size_t z = 0; z < dimension; ++z)
      {
        auto position = result->to_position(x, y, z);
        result->at(x, y, z) = kernel_f3(position);
      }
  return result;
}
#endif

#ifdef CUDA_RENDERING
GridF3<true>::grid_t make_density_grid(const GridInfo& info)
{
  auto generated = make_density_grid_aux(info);
#ifdef CUDA_GENERATION
  auto result = generated;
#else
  auto result = generated.copy_to_device();
#endif
  return result;
}
#else
GridF3<false>::grid_t make_density_grid(const GridInfo& info)
{
  auto generated = make_density_grid_aux(info);
#ifdef CUDA_GENERATION
  auto result = generated.copy_to_host();
#else
  auto result = generated;
#endif
  return result;
}
#endif
