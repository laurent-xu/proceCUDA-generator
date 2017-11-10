#include <app/make_density.hh>

void make_grids(const Camera& camera, bool& running,
                std::shared_ptr<std::vector<rendering::VerticesGrid>>* vertices)
{
  while (running)
  {
    // TODO Anatole Compute the grids_info according to the camera position
    //       with the octree
    std::vector<GridInfo> grids_info;
    for (size_t i = 0; i < 1; ++i)
      grids_info.emplace_back(1., F3::vec3_t(0., 0., 0.), 8);
    grids_info.emplace_back(1., F3::vec3_t(0., 0., 32.), 8);

    to_be_printed.clear();

    for (const auto& info: grids_info)
    {
      // TODO Anatole
      // if (cache_lru.contains(info))
      //    to_be_printed.push(cache_lru[info])
      //    continue;

      // GridF3<DensityTarget>::grid_t where DensityTarget depends on the
      // device used by the rendering
      auto density_grid = make_density_grid(info);

      auto hermitian_grid = rendering::HermitianGrid(density_grid, rendering::point_t(density_grid->dim_size()), 1);
      // Bellow the scale is 0.2. This value can be tweaked for bigger/smaller map
      auto vertices_grid = rendering::VerticesGrid(hermitian_grid, 0.2);
      // cache_lru.add(info, vertices_grids) TODO Anatole
      to_be_printed.push_back(vertices_grid);
    }
    // Display to_be_printed
    std::cerr << "Frame " << frame_idx++ << " is computed" << std::endl;

  }
}
