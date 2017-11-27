#include <density/F3Grid.hh>
#include <app/generation_kernel.hh>
#include <app/rendering.hh>
#include <app/make_density.hh>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <cuda_profiler_api.h>
#include <iostream>
#include <condition_variable>
#include <thread>
#include <random>
#include <cmath>

void print_help(const std::string& bin)
{
  std::cerr << "usage: " << bin << " grid_dim nb_thread_x nb_thread_y "
                                   "nb_thread_z max_grid_frame cache_size "
				   "max_grid_display "
                                   "is_real_time "
                                   "[x y z | nb_grids rand_min rand_max]"
                         << std::endl;
  std::cerr << "  grid_dim: the length of a density grid" << std::endl;
  std::cerr << "  nb_thread_[x|y|z]: the number of threads in a block. "
               "This has no consequences in CPU mode." << std::endl;
  std::cerr << "  max_grid_frame: the maximum number of new grids per frame"
            << std::endl;
  std::cerr << "  cache_size: the size of the cache containing the previous "
               "grids" << std::endl;
  std::cerr << "  is_real_time: It can be either true or false." << std::endl;
  std::cerr << "  x y z: In real time mode this is the initial position of "
               "the camera" << std::endl;
  std::cerr << "  nb_grids: Not in real time mode this is the number of camera "
               "positions to simulate" << std::endl;
  std::cerr << "  rand_min rand_max: Not in real mode, these values are the "
               "bounds of the position of the camera" << std::endl;

  std::exit(1);
}

int main(int argc, char* argv[])
{
  auto camera_position = glm::vec3();
  std::vector<glm::vec3> camera_positions;
  bool is_real_time = 0;
  size_t grid_dim = 0;
  size_t nb_thread_x = 0;
  size_t nb_thread_y = 0;
  size_t nb_thread_z = 0;
  size_t max_grid_per_frame = 0;
  size_t cache_size = 0;
  size_t max_grid_display = 0;

  std::random_device rd;
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

  int next_arg = 1;
  if (argc >= 7)
  {
    try
    {
      grid_dim = std::stoul(argv[next_arg++]);
      nb_thread_x = std::stoul(argv[next_arg++]);
      nb_thread_y = std::stoul(argv[next_arg++]);
      nb_thread_z = std::stoul(argv[next_arg++]);
      max_grid_per_frame = std::stoul(argv[next_arg++]);
      cache_size = std::stoul(argv[next_arg++]);
      max_grid_display = std::stoul(argv[next_arg++]);

      std::string is_real_time_str = argv[next_arg++];
      if (is_real_time_str != "false" && is_real_time_str != "true")
        print_help(argv[0]);
      is_real_time = is_real_time_str == "true";

      if (argc != next_arg + 3)
        print_help(argv[0]);
      if (is_real_time)
        for (size_t i = 0; i < 3; ++i)
          camera_position[i] = std::stod(argv[next_arg++]);
      else
      {
        auto nb_positions = std::stoul(argv[next_arg++]);
        double min = std::stod(argv[next_arg++]);
        double max = std::stod(argv[next_arg++]);
        std::uniform_real_distribution<> dis(min, max);
        for (size_t j = 0; j < nb_positions; ++j)
        {
          auto tmp_position = glm::vec3();
          for (size_t i = 0; i < 3; ++i)
            tmp_position[i] = dis(gen);
          camera_positions.push_back(tmp_position);
        }
      }
    }
    catch (std::exception e)
    {
      std::cerr << e.what() << std::endl;
      print_help(argv[0]);
    }
  }
  else
    print_help(argv[0]);

  auto grid_maker = AsynchronousGridMaker(grid_dim, nb_thread_x, nb_thread_y,
                                          nb_thread_z, max_grid_per_frame,
                                          cache_size, max_grid_display);

  if (is_real_time)
  {
    CERR << "Initial position is ("
         << camera_position.x << ", "
         << camera_position.y << ", "
         << camera_position.y << ")" << std::endl;

    auto generation_position = std::make_shared<glm::vec3>(camera_position);
    auto running = std::make_shared<bool>(true);
    std::shared_ptr<std::vector<std::shared_ptr<rendering::VerticesGrid>>> vertices;
    std::condition_variable cv_generation;
    std::mutex m;

    // Creating window
    auto window = std::make_shared<sf::Window>(sf::VideoMode(800, 600), argv[0],
                                               sf::Style::Default,
                                               sf::ContextSettings(32));
    window->setVerticalSyncEnabled(true);
    glEnable(GL_DEPTH_TEST);
    if (glewInit() == GLEW_OK)
      CERR << "Glew initialized successfully" << std::endl;

    auto renderer = AsynchronousRendering(window, &generation_position, running,
                                          &vertices, cv_generation, m);
    auto grid_maker_thread =
      grid_maker.make_grids_in_thread(&generation_position, running, &vertices,
                                      cv_generation, m);
    renderer.render_grids();
    {
      std::unique_lock<std::mutex> lock(m);
      CERR << "notify end " << std::endl;
      cv_generation.notify_one();
    }
    CERR << "notified" << std::endl;
    grid_maker_thread.join();
    CERR << "thread joined" << std::endl;
  }
  else
  {
    for (auto position: camera_positions)
      grid_maker.make_grid(position);
  }
  cudaProfilerStop();

  return 0;
}

