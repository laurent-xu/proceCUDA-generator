#include <app/rendering.hh>
#include <app/make_density.hh>
#include <mutex>
#include <thread>
#include <density/Sphere.hh>

bool AsynchronousRendering::update_position()
{
  bool moved = false;
  sf::Event event;
  auto deltaTime = clock.getElapsedTime().asSeconds() * 1000;
  deltaTime = 5.0f / 60.0f;
  // CERR << deltaTime << std::endl;
  while (window->pollEvent(event))
  {
    switch (event.type)
    {
      case sf::Event::Closed:
        *running = false;
        break;
      case sf::Event::KeyPressed:
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
        {
          *running = false;
          window->close();
          break;
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Z) ||
           sf::Keyboard::isKeyPressed(sf::Keyboard::W))
        {
          camera.processKeyboard(Camera_Movement::FORWARD, deltaTime);
          moved = true;
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::S))
        {
          camera.processKeyboard(Camera_Movement::BACKWARD, deltaTime);
          moved = true;
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Q) ||
           sf::Keyboard::isKeyPressed(sf::Keyboard::A))
        {
          camera.processKeyboard(Camera_Movement::LEFT, deltaTime);
          moved = true;
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::D))
        {
          camera.processKeyboard(Camera_Movement::RIGHT, deltaTime);
          moved = true;
        }
        break;

      case sf::Event::MouseMoved:
        auto new_position = sf::Mouse::getPosition();
        camera.processMouse(new_position.x - former_mouse.x,
                            former_mouse.y - new_position.y);
        former_mouse = new_position;
        moved = true;
    }
  }
  clock.restart();
  return moved;
}

void AsynchronousRendering::init_frame()
{
  // Clean the frame
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  shader.Use();
  GLint light_pos_loc = glGetUniformLocation(shader.getProgram(),
                                             "lightPos");
  GLint object_color_loc = glGetUniformLocation(shader.getProgram(),
                                                "objectColor");
  GLint light_color_loc  = glGetUniformLocation(shader.getProgram(),
                                                "lightColor");
  glUniform3f(object_color_loc, 1.0f, 0.5f, 0.31f);
  // Also set light's color (white)
  glUniform3f(light_color_loc,  1.0f, 1.0f, 1.0f);
  glUniform3f(light_pos_loc, light_pos.x, light_pos.y, light_pos.z);

  GLint model_loc = glGetUniformLocation(shader.getProgram(), "model");
  glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(model));
  GLint view_loc = glGetUniformLocation(shader.getProgram(), "view");
  glUniformMatrix4fv(view_loc, 1, GL_FALSE,
                     glm::value_ptr(camera.getViewMatrix()));
  GLint proj_loc = glGetUniformLocation(shader.getProgram(),
                                       "projection");
  glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm::value_ptr(projection));
  GLint viewer_loc_pos = glGetUniformLocation(shader.getProgram(),
                                            "viewerPos");
  glUniform3f(viewer_loc_pos, camera.getPosition().x, camera.getPosition().y,
              camera.getPosition().z);
}

using data_t = rendering::data_t;
using point_t = rendering::point_t;

void AsynchronousRendering::render_grids()
{
  // Note that we're translating the scene in the reverse direction of
  // where we want to move
  projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f,
                                             0.1f, 1000.0f);
  shader = Shader("../resources/shaders/vertex_shader.glsl",
                  "../resources/shaders/fragment_shader.glsl");

  glClearColor(0.1, 0.1, 0.1, 1.0);

  size_t frame_idx = 0;
  while (*running)
  {
    auto need_new_frame = update_position();
    auto current_position = camera.getPosition();
    // New vertices are generated if the position of the camera has changed
    if (need_new_frame)
    {
      auto new_position = std::make_shared<glm::vec3>(current_position);
      std::unique_lock<std::mutex> lock(m);
      std::atomic_store(generation_position, new_position);
      cv_generation.notify_one();
    }

    // If new vertices have been computed, we display them
    std::shared_ptr<std::vector<std::shared_ptr<rendering::VerticesGrid>>> tmp;
    tmp = std::atomic_exchange(vertices, tmp);
    if (tmp)
    {
      to_be_printed = tmp;
      need_new_frame = true;
    }

    if (need_new_frame && to_be_printed)
    {
      init_frame();
      //auto my_grids = grid_maker.make_grid(current_position, true);
      for (auto e : *to_be_printed)
      {
        e->initVAO();
        e->draw();
      }

      // Draw window
      window->display();
      CERR << "draw " << frame_idx++ << std::endl;
    }
  }
}
