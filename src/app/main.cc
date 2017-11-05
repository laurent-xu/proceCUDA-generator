#include <density/F3Grid.hh>
#include <app/generation_kernel.hh>

#include <iostream>
#include <GL/glew.h>
#include <rendering/hermitian-grid.hh>
#include <rendering/utils/nm-matrix.hpp>
#include <rendering/qr-decomposition.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <rendering/viewer/camera.hh>
#include <rendering/viewer/shader.hh>
#include <rendering/vertices-grid.hpp>

// Camera controls
void updateCamera(Camera &camera, float deltaTime, sf::Vector2f &formerPosition);

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

  // Creating window
  sf::Window *window = new sf::Window(sf::VideoMode(800, 600), "TestSphere",
                                      sf::Style::Default, sf::ContextSettings(32));
  window->setVerticalSyncEnabled(true);
  glEnable(GL_DEPTH_TEST);
  if (glewInit() == GLEW_OK)
    std::cout << "Glew initialized successfully" << std::endl;

  size_t frame_idx = 0;
  Camera camera;
  sf::Clock clock;

  bool running = true;
  std::vector<rendering::VerticesGrid> to_be_printed;
  while (running)
  {
    // Check if window is closed
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    sf::Event event;
    while (window->pollEvent(event))
      if (event.type == sf::Event::Closed)
        running = false;

    // Clean the frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // TODO Anatole Compute the grids_info according to the camera position
    //       with the octree
    std::vector<GridInfo> grids_info;
    for (size_t i = 0; i < 100; ++i)
      grids_info.emplace_back(1., F3::vec3_t(0., 0., 0.), 32);

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

    // Get input and update camera position
    sf::Vector2f mousePosition(sf::Mouse::getPosition());
    updateCamera(camera, clock.getElapsedTime().asSeconds(), mousePosition);
    clock.restart();

    // Draw window
    window->display();
  }
  delete window;
  return 0;
}

// Camera controls
void updateCamera(Camera &camera, float deltaTime, sf::Vector2f &formerPosition) {
  sf::Vector2f newPosition(sf::Mouse::getPosition());
  if(sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
    camera.processKeyboard(Camera_Movement::FORWARD, deltaTime);
  if(sf::Keyboard::isKeyPressed(sf::Keyboard::S))
    camera.processKeyboard(Camera_Movement::BACKWARD, deltaTime);
  if(sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
    camera.processKeyboard(Camera_Movement::LEFT, deltaTime);
  if(sf::Keyboard::isKeyPressed(sf::Keyboard::D))
    camera.processKeyboard(Camera_Movement::RIGHT, deltaTime);
  camera.processMouse(newPosition.x - formerPosition.x,
                      formerPosition.y - newPosition.y);
  formerPosition = newPosition;
}
