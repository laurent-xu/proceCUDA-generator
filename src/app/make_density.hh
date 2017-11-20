#pragma once
#include <rendering/viewer/camera.hh>
#include <rendering/vertices-grid.hpp>
#include <app/generation_kernel.hh>
#include <utils/glm.hh>
#include <condition_variable>
#include <thread>
#include <memory>
#include <vector>


class AsynchronousGridMaker
{
public:
  AsynchronousGridMaker(size_t nb_voxels, size_t nb_thread_x,
                        size_t nb_thread_y, size_t nb_thread_z)
    : nb_voxels(nb_voxels),
      nb_thread_x(nb_thread_x),
      nb_thread_y(nb_thread_y),
      nb_thread_z(nb_thread_z)
  {
  }


  void make_octree(const glm::vec3& position);

  std::shared_ptr<std::vector<rendering::VerticesGrid>>
  make_grid(const glm::vec3& position);

  void make_grids(std::shared_ptr<glm::vec3>* generation_position,
                  bool* running,
                  std::shared_ptr<std::vector<rendering::VerticesGrid>>*
                    vertices,
                  std::condition_variable& cv_generation,
                  std::mutex& m);

  std::thread make_grids_in_thread(std::shared_ptr<glm::vec3>*
                                   generation_position,
                                   std::shared_ptr<bool> running,
                                   std::shared_ptr<std::vector<
                                                   rendering::VerticesGrid>>*
                                    vertices,
                                   std::condition_variable& cv_generation,
                                   std::mutex& m)
  {
    return std::thread([&](){make_grids(generation_position, running.get(),
                                        vertices, cv_generation, m);});
  }

private:
  size_t nb_voxels;
  size_t nb_thread_x;
  size_t nb_thread_y;
  size_t nb_thread_z;
  std::vector<GridInfo> grids_info;
};

#ifdef CUDA_GENERATION
static inline GridF3<true>::grid_t make_density_grid_aux(const GridInfo& info,
                                                         size_t nb_thread_x,
                                                         size_t nb_thread_y,
                                                         size_t nb_thread_z)
{
  size_t dimension = info.dimension;
  size_t block_dim_x = (dimension + nb_thread_x - 1) / nb_thread_x;
  size_t block_dim_y = (dimension + nb_thread_y - 1) / nb_thread_y;
  size_t block_dim_z = (dimension + nb_thread_z - 1) / nb_thread_z;
  dim3 Dg(block_dim_x, block_dim_y, block_dim_z);
  dim3 Db(nb_thread_x, nb_thread_y, nb_thread_z);

  auto result = GridF3<true>::get_grid(info);
  result->hold();
  kernel_f3_caller<<<Dg,Db>>>(*result);
  // TODO Remove the next line
  cudaDeviceSynchronize();
  result->release();
  return result;
}
#else
static inline GridF3<false>::grid_t make_density_grid_aux(const GridInfo& info,
                                                          size_t, size_t,
                                                          size_t)
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
static inline GridF3<true>::grid_t make_density_grid(const GridInfo& info,
                                                     size_t nb_thread_x,
                                                     size_t nb_thread_y,
                                                     size_t nb_thread_z)
{
  auto generated = make_density_grid_aux(info, nb_thread_x, nb_thread_y, nb_thread_z);
#ifdef CUDA_GENERATION
  auto result = generated;
#else
  auto result = copy_to_device(generated);
#endif
  return result;
}
#else
static inline GridF3<false>::grid_t make_density_grid(const GridInfo& info,
                                                      size_t nb_thread_x,
                                                      size_t nb_thread_y,
                                                      size_t nb_thread_z)
{
  auto generated = make_density_grid_aux(info, nb_thread_x, nb_thread_y,
                                         nb_thread_z);
#ifdef CUDA_GENERATION
  auto result = copy_to_host(generated);
#else
  auto result = generated;
#endif
  return result;
}
#endif

void make_grids(const Camera& camera, bool& running,
                std::shared_ptr<std::vector<rendering::VerticesGrid>>*
                vertices);
