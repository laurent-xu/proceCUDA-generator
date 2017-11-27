#include <app/make_density.hh>
#include <app/generation_kernel.hh>

// This function must not return grids that overlap, it should return at least
// the grid containing the camera with the lowest possible precision, denoted
// initial_precision. Unless if this position is in the cache.
// It is supposed to return  max_grid_per_frame grids that are not in the cache
// If the nearest cache_size grids are already in the cache, the function won't
// return anything

const double AsynchronousGridMaker::vector[7][3] =
{
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 1, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

void AsynchronousGridMaker::make_octree(const glm::vec3& position)
{
  grids_info.clear();
  double frequency = 1.;
  size_t it = 0, nb_grid = 1;
  int coefficient = 1;
  GridInfo::vec3_t new_position;
  new_position.x = position.x / (nb_voxels * frequency);
  new_position.y = position.y / (nb_voxels * frequency);
  new_position.z = position.z / (nb_voxels * frequency);
  grids_info.emplace_back(frequency, new_position, nb_voxels);
  while (nb_grid < max_grid_per_frame)
  {
    GridInfo::vec3_t new_position;
    new_position.x = position.x / (nb_voxels * frequency)
        + coefficient * AsynchronousGridMaker::vector[it][0];
    new_position.y = position.y / (nb_voxels * frequency)
        + coefficient * AsynchronousGridMaker::vector[it][1];
    new_position.z = position.z / (nb_voxels * frequency)
        + coefficient * AsynchronousGridMaker::vector[it][2];
    grids_info.emplace_back(frequency, new_position, nb_voxels);
    ++it; 
    if (it == 7)
    {
      frequency *= 2;
      it = 0;
      coefficient = -coefficient;
    }
    ++nb_grid;    
  }
}

std::shared_ptr<std::vector<std::shared_ptr<rendering::VerticesGrid>>>
AsynchronousGridMaker::make_grid(const glm::vec3& position, bool render)
{
  size_t nb_curr_gen = 0, nb_return = 0;
  make_octree(position);
  auto to_be_printed =
    std::make_shared<std::vector<std::shared_ptr<rendering::VerticesGrid>>>();
  static std::vector<GridInfo> generation_grids_info;
  static std::vector<GridF3<false>::grid_t> density_grids;

  for (const auto& info: grids_info)
  {
    if (nb_return == cache_size || nb_curr_gen == max_grid_per_frame)
      break;
    ++nb_return;
    if (cache_lru.contains(info))
      to_be_printed->push_back(cache_lru.get(info));
    else
    {
      ++nb_curr_gen;
      generation_grids_info.push_back(info);
    }
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
    cache_lru.add(density_grid->get_grid_info(), vertices_grid);
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

    CERR << "Compute" << std::endl;
    size_t last_nb_grids = 0;

    while (true)
    {
      if (!current_position)
        current_position = std::make_shared<glm::vec3>(previous_position);
      if (previous_position == *current_position &&
          last_nb_grids == cache_lru.size())
        break;
      previous_position = *current_position;
      make_octree(previous_position);
      auto to_be_printed = make_grid(previous_position, true);
      last_nb_grids = to_be_printed->size();

      CERR << "Frame " << frame_idx++ << " is computed" << std::endl;
      std::atomic_store(vertices, to_be_printed);
      current_position = std::atomic_exchange(generation_position,
                                              current_position);
    }
  }
  CERR << "End of make grids" << std::endl;
}
