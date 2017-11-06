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
#include <iostream>
#include <vector>

#ifdef CUDA_GENERATION
static inline GridF3<true>::grid_t make_density_grid_aux(const GridInfo& info)
{
  size_t dimension = info.dimension;
  size_t thread_per_dim = 8;
  size_t block_dim = (dimension + thread_per_dim - 1) / thread_per_dim;
  dim3 Dg(block_dim, block_dim, block_dim);
  dim3 Db(thread_per_dim, thread_per_dim, thread_per_dim);

  auto result = GridF3<true>::get_grid(info);
  result->hold();
  kernel_f3_caller<<<Dg,Db>>>(*result);
  // TODO Remove the next line
  cudaDeviceSynchronize();
  result->release();
  return result;
}
#else
static inline GridF3<false>::grid_t make_density_grid_aux(const GridInfo& info)
{
  size_t dimension = info.dimension;
  auto result = GridF3<false>::get_grid(info);
  for (size_t x = 0; x < dimension; ++x)
    for (size_t y = 0; y < dimension; ++y)
      for (size_t z = 0; z < dimension; ++z)
      {
        auto position = result->to_position(x, y, z);
        result->at(x, y, z) = kernel_f3(position);
      }
  return result;
}
#endif

#ifdef CUDA_RENDERING
static inline GridF3<true>::grid_t make_density_grid(const GridInfo& info)
{
  auto generated = make_density_grid_aux(info);
#ifdef CUDA_GENERATION
  auto result = generated;
#else
  auto result = copy_to_device(generated);
#endif
  return result;
}
#else
static inline GridF3<false>::grid_t make_density_grid(const GridInfo& info)
{
  auto generated = make_density_grid_aux(info);
#ifdef CUDA_GENERATION
  auto result = copy_to_host(generated);
#else
  auto result = generated;
#endif
  return result;
}
#endif

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
  glm::mat4 model;
  glm::mat4 view;
  glm::vec3 lightPos(-4.0f, -13.0f, 9.0f);
  // Note that we're translating the scene in the reverse direction of where we want to move
  view = glm::translate(view, glm::vec3(0.0f, 0.0f, 0.0f));
  glm::mat4 projection;
  projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f);
  Shader shader("../resources/shaders/vertex_shader.glsl",
                "../resources/shaders/fragment_shader.glsl");
  sf::Vector2f mousePosition(sf::Mouse::getPosition());

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
    for (size_t i = 0; i < 1; ++i)
      grids_info.emplace_back(1., F3::vec3_t(0., 0., 0.), 32);
    grids_info.emplace_back(1., F3::vec3_t(0., 0., 32.), 32);

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
    updateCamera(camera, clock.getElapsedTime().asSeconds(), mousePosition);
    clock.restart();

    shader.Use();

    GLint lightPosLoc = glGetUniformLocation(shader.getProgram(), "lightPos");
    GLint objectColorLoc = glGetUniformLocation(shader.getProgram(), "objectColor");
    GLint lightColorLoc  = glGetUniformLocation(shader.getProgram(), "lightColor");
    glUniform3f(objectColorLoc, 1.0f, 0.5f, 0.31f);
    glUniform3f(lightColorLoc,  1.0f, 1.0f, 1.0f); // Also set light's color (white)
    glUniform3f(lightPosLoc, lightPos.x, lightPos.y, lightPos.z);

    GLint modelLoc = glGetUniformLocation(shader.getProgram(), "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    GLint viewLoc = glGetUniformLocation(shader.getProgram(), "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(camera.getViewMatrix()));
    GLint projLoc = glGetUniformLocation(shader.getProgram(), "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    GLint viewerLocPos = glGetUniformLocation(shader.getProgram(), "viewerPos");
    glUniform3f(viewerLocPos, camera.getPosition().x, camera.getPosition().y,
                camera.getPosition().z);

    for (auto &e : to_be_printed)
      e.draw();

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
