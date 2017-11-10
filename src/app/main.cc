#include <density/F3Grid.hh>
#include <app/generation_kernel.hh>
#include <app/make_density.hh>

#include <iostream>

void print_help(const std::string& bin)
{
  std::cerr << "usage: " << bin << " x y z" << std::endl;
  std::exit(1);
}

F3::vec3_t parse_camera(int argc, char* argv[])
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

  std::cout << "Initial position is ("
            << camera_position.x << ", "
            << camera_position.y << ", "
            << camera_position.y << ")" << std::endl;

  return camera_position;
}

int main(int argc, char* argv[])
{
  camera_position = parse_camera();
  auto camera = Camera(camera_position);
  bool running = true;
  bool moved = true;
  std::condition_variable cv_rendering;
  std::condition_variable cv_generation;
  std::shared_ptr<std::vector<rendering::VerticesGrid>> vertices;
  sf::Vector2f former_position;

  // Creating window
  sf::Window *window = new sf::Window(sf::VideoMode(800, 600), argv[0],
                                      sf::Style::Default,
                                      sf::ContextSettings(32));
  window->setVerticalSyncEnabled(true);
  glEnable(GL_DEPTH_TEST);
  std::shared_ptr<std::vector<rendering::VerticesGrid>>* to_be_printed;
  if (glewInit() == GLEW_OK)
    std::cout << "Glew initialized successfully" << std::endl;

  auto generation_thread = std::thread(make_grids, camera, running,
                                       to_be_printed, cv_generation,
                                       cv_rendering);
  auto rendering_thread = std::thread(render_grids, camera, window, running,
                                      to_be_printed, cv_rendering);
  sf::Clock clock;
  while (running)
  {
    sf::Event event;
    auto position = camera.position;
    auto deltaTime = clock.getElapsedTime().asSeconds();
    while (window->pollEvent(event))
    {
      switch (event.type)
      {
        case sf::Event::Closed:
          running = false;
          break;
        case sf::Event::KeyPressed:
          if(sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
          {
            camera.processKeyboard(Camera_Movement::FORWARD, deltaTime);
            moved = true;
          }

          if(sf::Keyboard::isKeyPressed(sf::Keyboard::S))
          {
            camera.processKeyboard(Camera_Movement::BACKWARD, deltaTime);
            moved = true;
          }

          if(sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
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
    if (moved)
      cv_rendering.notify_one();
    if (position != camera.position)
      cv_generation.notify_one();
      
  }

  rendering_thread.join();
  generation_thread.join();
  delete window;

  return 0;
}

