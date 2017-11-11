#pragma once
#include <rendering/viewer/camera.hh>
#include <rendering/vertices-grid.hpp>
#include <app/generation_kernel.hh>
#include <condition_variable>
#include <thread>


class AsynchronousGridMaker
{
public:
  AsynchronousGridMaker(std::shared_ptr<glm::vec3>* generation_position,
                        std::shared_ptr<bool> running,
                        std::shared_ptr<std::vector<rendering::VerticesGrid>>* vertices,
                        std::condition_variable& cv_generation,
                        std::mutex& m,
                        size_t nb_voxels)
    : generation_position(generation_position),
      running(running),
      vertices(vertices),
      cv_generation(cv_generation),
      m(m),
      nb_voxels(nb_voxels)
  {
  }


  void make_octree(const glm::vec3& position);

  void make_grids();

  std::thread make_grids_in_thread()
  {
    return std::thread([&](){make_grids();});
  }

private:
  std::shared_ptr<glm::vec3>* generation_position;
  std::shared_ptr<bool> running;
  std::shared_ptr<std::vector<rendering::VerticesGrid>>* vertices;
  std::condition_variable& cv_generation;
  std::mutex& m;
  size_t nb_voxels;
  std::vector<GridInfo> grids_info;
  glm::vec3 previous_position;
};

#ifdef CUDA_GENERATION
static inline GridF3<true>::grid_t make_density_grid_aux(const GridInfo& info)
{
  size_t dimension = info.dimension;
  size_t thread_per_dim = 4;
  size_t block_dim = (dimension + thread_per_dim - 1) / thread_per_dim;
  dim3 Dg(block_dim, block_dim, block_dim);
  dim3 Db(thread_per_dim, thread_per_dim, thread_per_dim);

  auto result = GridF3<true>::get_grid(info);
  result->hold();
  kernel_f3_caller<<<Dg,Db>>>(*result);
  // TODO Remove the next line
  cudaDeviceSynchronize();
  result->release();
  return result;
}
#else
static inline GridF3<false>::grid_t make_density_grid_aux(const GridInfo& info)
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
static inline GridF3<true>::grid_t make_density_grid(const GridInfo& info)
{
  auto generated = make_density_grid_aux(info);
#ifdef CUDA_GENERATION
  auto result = generated;
#else
  auto result = copy_to_device(generated);
#endif
  return result;
}
#else
static inline GridF3<false>::grid_t make_density_grid(const GridInfo& info)
{
  auto generated = make_density_grid_aux(info);
#ifdef CUDA_GENERATION
  auto result = copy_to_host(generated);
#else
  auto result = generated;
#endif
  return result;
}
#endif

void make_grids(const Camera& camera, bool& running,
                std::shared_ptr<std::vector<rendering::VerticesGrid>>* vertices);
