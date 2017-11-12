#include <app/make_density.hh>
#include <app/generation_kernel.hh>

void AsynchronousGridMaker::make_octree(const glm::vec3& position)
{
  grids_info.clear();
  GridInfo::vec3_t origin;
  origin.x = position.x / nb_voxels;
  origin.y = position.y / nb_voxels;
  origin.z = position.z / nb_voxels;
  for (auto x: {-1, 0, 1})
    for (auto y: {-1, 0, 1})
      for (auto z: {-1, 0, 1})
        grids_info.emplace_back(1., origin + GridInfo::vec3_t(x, y, z),
                                nb_voxels);
}

void AsynchronousGridMaker::make_grids()
{
  size_t frame_idx = 0;
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
    auto to_be_printed =
      std::make_shared<std::vector<rendering::VerticesGrid>>();
    std::vector<GridInfo> generation_grids_info;

    for (const auto& info: grids_info)
    {
      // TODO Anatole
      // if (cache_lru.contains(info))
      //    to_be_printed.push(cache_lru[info])
      // else
      generation_grids_info.push_back(info);
    }

    for (const auto& info: grids_info)
    {
      // GridF3<DensityTarget>::grid_t where DensityTarget depends on the
      // device used by the rendering
      auto density_grid = make_density_grid(info);

      auto hermitian_grid = rendering::HermitianGrid(density_grid,
                                                     rendering::point_t(density_grid->dim_size()),
                                                     1);
      // Bellow the scale is 0.2. This value can be tweaked for bigger/smaller map
      auto vertices_grid = rendering::VerticesGrid(hermitian_grid, 0.2);
      // cache_lru.add(info, vertices_grids) TODO Anatole
      to_be_printed->push_back(vertices_grid);
    }
    CERR << "Frame " << frame_idx++ << " is computed" << std::endl;
    std::atomic_store(vertices, to_be_printed);
  }
  CERR << "End of make grids" << std::endl;
}
