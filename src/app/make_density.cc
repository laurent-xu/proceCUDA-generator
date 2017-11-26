#include <app/make_density.hh>
#include <app/generation_kernel.hh>

// This function must not return grids that overlap, it should return at least
// the grid containing the camera with the lowest possible precision, denoted
// initial_precision. Unless if this position is in the cache.
// It is supposed to return  max_grid_per_frame grids that are not in the cache
// If the nearest cache_size grids are already in the cache, the function won't
// return anything
void AsynchronousGridMaker::make_octree(const glm::vec3& position)
{
  grids_info.clear();
  // TODO Anatole below
  GridInfo::vec3_t origin;
  double initial_precision = 1.;

  // The next three lines give the offset of the grid containing the camera with
  // the given precision.
  origin.x = position.x / (nb_voxels * initial_precision);
  origin.y = position.y / (nb_voxels * initial_precision);
  origin.z = position.z / (nb_voxels * initial_precision);

  for (auto x = -2; x < 2; ++x)
    for (auto y = -2; y < 2; ++y)
      for (auto z = -2; z < 2; ++z)
        grids_info.emplace_back(1., origin + GridInfo::vec3_t(x, y, z),
                                nb_voxels);
}

std::shared_ptr<std::vector<std::shared_ptr<rendering::VerticesGrid>>>
AsynchronousGridMaker::make_grid(const glm::vec3& position, bool render)
{
  make_octree(position);
  auto to_be_printed =
    std::make_shared<std::vector<std::shared_ptr<rendering::VerticesGrid>>>();
  static std::vector<GridInfo> generation_grids_info;
  static std::vector<GridF3<false>::grid_t> density_grids;

  for (const auto& info: grids_info)
  {
    // TODO Anatole
    // if (cache_lru.contains(info))
    //    to_be_printed.push(cache_lru[info])
    // else
    generation_grids_info.push_back(info);
  }

  for (const auto& info: generation_grids_info)
  {
    // GridF3<DensityTarget>::grid_t where DensityTarget depends on the
    // device used by the rendering
    density_grids.push_back(make_density_grid(info, nb_thread_x, nb_thread_y,
                                              nb_thread_z));
  }

  for (auto& density_grid: density_grids)
  {
    auto hermitian_grid =
      rendering::HermitianGrid(density_grid,
                               rendering::point_t(density_grid->dim_size()),
                               1);
    auto vertices_grid =
      std::make_shared<rendering::VerticesGrid>(hermitian_grid, 1);
    // cache_lru.add(info, vertices_grids) TODO Anatole
    to_be_printed->push_back(vertices_grid);
    density_grid->release();
  }
  generation_grids_info.clear();
  density_grids.clear();
  return to_be_printed;
}

void
AsynchronousGridMaker::make_grids(std::shared_ptr<glm::vec3>*
                                  generation_position,
                                  bool* running,
                                  std::shared_ptr<std::vector<
                                     std::shared_ptr<rendering::VerticesGrid>>>*
                                     vertices,
                                  std::condition_variable& cv_generation,
                                  std::mutex& m)
{
  size_t frame_idx = 0;
  glm::vec3 previous_position;

  while (*running)
  {
    std::shared_ptr<glm::vec3> current_position;
    {
      std::unique_lock<std::mutex> lock(m);
      while(!*generation_position && *running)
        cv_generation.wait(lock);
      current_position = std::atomic_exchange(generation_position,
                                              current_position);
    }

    if (!current_position || previous_position == *current_position)
      continue;
    previous_position = *current_position;

    CERR << "Compute" << std::endl;

    // TODO Anatole Compute the grids_info according to the camera position
    //       with the octree
    make_octree(*current_position);
    auto to_be_printed = make_grid(previous_position, true);
    make_octree(*current_position);

    CERR << "Frame " << frame_idx++ << " is computed" << std::endl;
    std::atomic_store(vertices, to_be_printed);
  }
  CERR << "End of make grids" << std::endl;
}
