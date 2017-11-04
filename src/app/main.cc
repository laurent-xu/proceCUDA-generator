#include <density/F3Grid.hh>
#include <app/generation_kernel.hh>
#include <iostream>
#include <vector>

void print_help(const std::string& bin)
{
  std::cerr << "usage: " << bin << " x y z" << std::endl;
  std::exit(1);
}

int main(int argc, char* argv[])
{
  auto camera_position = F3::vec3_t();
  if (argc == 4)
  {
    size_t i = 1;
    try
    {
      for (; i < 4; ++i)
        camera_position[i - 1] = std::stod(argv[i]);
    }
    catch (std::exception e)
    {
      std::cerr << e.what() << std::endl;
      print_help(argv[0]);
    }
  }
  else
    print_help(argv[0]);

  std::cerr << "Initial position is ("
            << camera_position.x << ", "
            << camera_position.y << ", "
            << camera_position.y << ")" << std::endl;

  // Aurelien Init the OpenGL frame here

  size_t frame_idx = 0;
  while (true)
  {
    // TODO Aurelien Clean the frame
    // TODO Anatole Compute the grids_info according to the camera position
    //       with the octree
    std::vector<GridInfo> grids_info;
    for (size_t i = 0; i < 100; ++i)
      grids_info.emplace_back(1., F3::vec3_t(0., 0., 0.), 32);

    // to_be_printed.clear()
    for (const auto& info: grids_info)
    {
      // TODO Anatole
      // if (cache_lru.contains(info))
      //    to_be_printed.push(cache_lru[info])
      //    continue;

      // GridF3<DensityTarget>::grid_t where DensityTarget depends on the
      // device used by the rendering
      auto density_grid = make_density_grid(info);

      // auto hermitian_grid = HermitianGrid(density_grid); TODO Aurelien
      // auto vertices_grid = MakeVertices(hermitian_grid); TODO Aurelien
      // cache_lru.add(info, vertices_grids) TODO Anatole
      // to_be_printed.push(vertices_grids) TODO Aurelien
    }
    // Display to_be_printed
    std::cerr << "Frame " << frame_idx++ << " is computed" << std::endl;
    // TODO Aurelien wait for input
    // TODO Aurelien update camera position
  }
  return 0;
}
