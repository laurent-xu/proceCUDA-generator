#pragma once
#include <rendering/viewer/camera.hh>
#include <rendering/vertices-grid.hpp>
#include <app/generation_kernel.hh>
#include <octree/lru.hh>
#include <utils/glm.hh>
#include <utils/cudamacro.hh>
#include <condition_variable>
#include <thread>
#include <memory>
#include <vector>
#include <cmath>


class InfoHash {
  public:
  std::size_t operator()(GridInfo const& c) const {
      size_t h1 = std::hash<double>()(c.offset.x);
      size_t h2 = std::hash<double>()(c.offset.y);
      size_t h3 = std::hash<double>()(c.offset.z);
      size_t h4 = std::hash<double>()(c.dimension);
      size_t h5 = std::hash<size_t>()(c.precision);
      return (((h1 ^ (h2 << 1)) ^ h3) ^ (h4 << 1)) ^ h5;
  }
};

class AsynchronousGridMaker
{
public:
  AsynchronousGridMaker(size_t nb_voxels, size_t nb_thread_x,
                        size_t nb_thread_y, size_t nb_thread_z,
                        size_t max_grid_per_frame, size_t cache_size,
                        size_t max_grid_display, size_t nb_streams)
    : nb_voxels(nb_voxels),
      nb_thread_x(nb_thread_x),
      nb_thread_y(nb_thread_y),
      nb_thread_z(nb_thread_z),
      max_grid_per_frame(max_grid_per_frame),
      cache_size(cache_size),
      max_grid_display(max_grid_display),
      cache_lru(cache_size),
      nb_streams(nb_streams)
  {
  }


  void make_octree(const glm::vec3& position,
                   std::shared_ptr<std::vector<std::shared_ptr<rendering::VerticesGrid>>>
                   to_be_printed, std::vector<GridInfo>& generation_grids_info);

  std::shared_ptr<std::vector<std::shared_ptr<rendering::VerticesGrid>>>
  make_grid(const glm::vec3& position, bool render = false);

  void make_grids(std::shared_ptr<glm::vec3>* generation_position,
                  bool* running,
                  std::shared_ptr<std::vector<std::shared_ptr<rendering::VerticesGrid>>>*
                    vertices,
                  std::condition_variable& cv_generation,
                  std::mutex& m);

  std::thread make_grids_in_thread(std::shared_ptr<glm::vec3>*
                                   generation_position,
                                   std::shared_ptr<bool> running,
                                   std::shared_ptr<std::vector<
                                                   std::shared_ptr<rendering::VerticesGrid>>>*
                                    vertices,
                                   std::condition_variable& cv_generation,
                                   std::mutex& m)
  {
    return std::thread([&](){make_grids(generation_position, running.get(),
                                        vertices, cv_generation, m);});
  }
  static const int vector[7][3];

private:
  size_t nb_voxels;
  size_t nb_thread_x;
  size_t nb_thread_y;
  size_t nb_thread_z;
  size_t max_grid_per_frame;
  size_t cache_size;
  size_t max_grid_display;
  LRUCache<GridInfo, std::shared_ptr<rendering::VerticesGrid>, InfoHash>
    cache_lru;
  std::vector<GridInfo> grids_info;
  bool done_generation;
  size_t nb_streams;
};

#ifdef CUDA_GENERATION
static inline GridF3<false>::grid_t make_density_grid_aux(const GridInfo& info,
                                                          size_t nb_thread_x,
                                                          size_t nb_thread_y,
                                                          size_t nb_thread_z,
                                                          size_t stream_idx,
                                                          size_t nb_streams)
{
  static cudaStream_t* streams;
  static bool initialized = false;
  if (!initialized)
  {
    initialized = true;
    streams = new cudaStream_t[nb_streams];
    for (size_t i = 0; i < nb_streams; ++i)
      cudaStreamCreate(&streams[i]);
  }

  size_t dimension = info.dimension;
  size_t block_dim_x = (dimension + nb_thread_x - 1) / nb_thread_x;
  size_t block_dim_y = (dimension + nb_thread_y - 1) / nb_thread_y;
  size_t block_dim_z = (dimension + nb_thread_z - 1) / nb_thread_z;
  dim3 Dg(block_dim_x, block_dim_y, block_dim_z);
  dim3 Db(nb_thread_x, nb_thread_y, nb_thread_z);

  auto result_d = GridF3<true>::get_grid(info);
  result_d->hold();
  auto& stream = streams[stream_idx];
  kernel_f3_caller<<<Dg,Db, 0, stream>>>(*result_d);
  return copy_to_host_async(result_d, stream);
}
#else
static inline GridF3<false>::grid_t make_density_grid_aux(const GridInfo& info,
                                                          size_t, size_t,
                                                          size_t, size_t,
                                                          size_t)
{
  size_t dimension = info.dimension;
  auto result = GridF3<false>::get_grid(info);
  #pragma omp parallel for
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

static inline GridF3<false>::grid_t make_density_grid(const GridInfo& info,
                                                      size_t nb_thread_x,
                                                      size_t nb_thread_y,
                                                      size_t nb_thread_z,
                                                      size_t stream_idx,
                                                      size_t nb_streams)
{
  return make_density_grid_aux(info, nb_thread_x, nb_thread_y,
                               nb_thread_z, stream_idx, nb_streams);
}
