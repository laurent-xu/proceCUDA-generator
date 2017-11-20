#include <density/F3Grid.hh>
#include <app/generation_kernel.hh>
#include <app/rendering.hh>
#include <app/make_density.hh>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <condition_variable>
#include <thread>
#include <cuda_profiler_api.h>

void print_help(const std::string& bin)
{
  std::cerr << "usage: " << bin << " grid_dim nb_thread_x nb_thread_y "
                                   "nb_thread_z is_real_time "
                                   "x1 y1 z1 [x2 y2 z2...] " << std::endl;
  std::cerr << "  grid_dim: the length of a density grid" << std::endl;
  std::cerr << "  nb_thread_[x|y|z]: the number of threads in a block. "
               "This has no consequences in CPU mode." << std::endl;
  std::cerr << "  is_real_time: It can be either true or false." << std::endl;
  std::cerr << "  x1 y1 z1: In real time mode this is the initial position of "
               "the camera" << std::endl;
  std::cerr << "  x1 y1 z1 [x2 y2 z2...] is the list of the positions of the "
               "camera" << std::endl;

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

  if (argc >= 6)
  {
    try
    {
      grid_dim = std::stoul(argv[1]);
      nb_thread_x = std::stoul(argv[2]);
      nb_thread_y = std::stoul(argv[3]);
      nb_thread_z = std::stoul(argv[4]);

      std::string is_real_time_str = argv[5];
      if (is_real_time_str != "false" && is_real_time_str != "true")
        print_help(argv[0]);
      is_real_time = is_real_time_str == "true";

      if (is_real_time)
      {
        if (argc != 9)
          print_help(argv[0]);
        for (size_t i = 0; i < 3; ++i)
          camera_position[i] = std::stod(argv[i + 6]);
      }
      else
      {
        if (argc % 3 != 0)
          print_help(argv[0]);
        else
          for (int last_parsed = 6; last_parsed < argc; last_parsed += 3)
          {
            auto tmp_position = glm::vec3();
            for (size_t i = 0; i < 3; ++i)
              tmp_position[i] = std::stod(argv[i + last_parsed]);
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
                                          nb_thread_z);

  if (is_real_time)
  {
    CERR << "Initial position is ("
         << camera_position.x << ", "
         << camera_position.y << ", "
         << camera_position.y << ")" << std::endl;

    auto generation_position = std::make_shared<glm::vec3>(camera_position);
    auto running = std::make_shared<bool>(true);
    std::shared_ptr<std::vector<rendering::VerticesGrid>> vertices;
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

