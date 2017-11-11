#include <app/make_density.hh>

void AsynchronousGridMaker::make_octree(std::shared_ptr<glm::vec3> position)
{
  grids_info.clear();
  size_t nb_voxels = 32.;
  F3::vec3_t origin;
  origin.x = int(position.x / nb_voxels) * nb_voxels;
  origin.y = int(position.y / nb_voxels) * nb_voxels;
  origin.z = int(position.z / nb_voxels) * nb_voxels;
  for (auto x: {-1, 0, 1})
    for (auto y: {-1, 0, 1})
      for (auto z: {-1, 0, 1})
        grids_info.emplace_back(1., position + F3::vec3_t(x, y, z) * nb_voxels,
                                nb_voxels);
}

void AsynchronousGridMaker::make_grids()
{
  while (running)
  {
    while(!*generation_position)
      cv_generation(lock);
    std::shared_ptr<glm::vec3> current_position;
    std::atomic_exchange(generation_position, current_position);

    // TODO Anatole Compute the grids_info according to the camera position
    //       with the octree
    make_octree(current_position);
    std::vector<rendering::VerticesGrid> to_be_printed;
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
      to_be_printed.push_back(vertices_grid);
    }
    std::cerr << "Frame " << frame_idx++ << " is computed" << std::endl;
    std::atomic_load(vertices, to_be_printed);

  }
}
