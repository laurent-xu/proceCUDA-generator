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
  std::cerr << "usage: " << bin << " x y z" << std::endl;
  std::exit(1);
}

glm::vec3 parse_camera(int argc, char* argv[])
{
  auto camera_position = glm::vec3();
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

  CERR << "Initial position is ("
       << camera_position.x << ", "
       << camera_position.y << ", "
       << camera_position.y << ")" << std::endl;

  return camera_position;
}

int main(int argc, char* argv[])
{
  auto generation_position =
    std::make_shared<glm::vec3>(parse_camera(argc, argv));
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
  auto grid_maker = AsynchronousGridMaker(&generation_position, running,
                                          &vertices, cv_generation, m, 32);

  auto grid_maker_thread = grid_maker.make_grids_in_thread();
  renderer.render_grids();
  {
    std::unique_lock<std::mutex> lock(m);
    CERR << "notify end " << std::endl;
    cv_generation.notify_one();
  }
  CERR << "notified" << std::endl;
  grid_maker_thread.join();
  std::cerr << "thread joined" << std::endl;
  cudaProfilerStop();

  return 0;
}

