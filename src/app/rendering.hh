#pragma once
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <GL/glew.h>
#include <rendering/hermitian-grid.hh>
#include <rendering/utils/nm-matrix.hpp>
#include <rendering/qr-decomposition.hpp>
#include <rendering/viewer/camera.hh>
#include <rendering/viewer/shader.hh>
#include <rendering/vertices-grid.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <condition_variable>

class AsynchronousRendering
{
public:
  AsynchronousRendering(const std::shared_ptr<sf::Window>& window,
                        std::shared_ptr<glm::vec3>* generation_position,
                        std::shared_ptr<bool> running,
                        std::shared_ptr<std::vector<rendering::VerticesGrid>>* vertices,
                        std::condition_variable& cv_generation,
                        std::mutex& m)
    : window(window),
      camera(**generation_position),
      generation_position(generation_position),
      running(running),
      vertices(vertices),
      cv_generation(cv_generation),
      m(m),
      light_pos(glm::vec3(-4.0f, -13.0f, 9.0f)),
      former_mouse(sf::Mouse::getPosition()),
      shader("../resources/shaders/vertex_shader.glsl",
             "../resources/shaders/fragment_shader.glsl")
  {
  }

  void render_grids();

private:
  bool update_position();
  void init_frame();

  std::shared_ptr<sf::Window> window;
  Camera camera;
  std::shared_ptr<glm::vec3>* generation_position;
  std::shared_ptr<bool> running;
  std::shared_ptr<std::vector<rendering::VerticesGrid>>* vertices;
  std::shared_ptr<std::vector<rendering::VerticesGrid>> to_be_printed;
  std::condition_variable& cv_generation;
  std::mutex& m;
  sf::Clock clock;
  glm::mat4 model;
  glm::mat4 view;
  glm::vec3 light_pos;
  glm::mat4 projection;
  sf::Vector2<int> former_mouse;
  Shader shader;
};
