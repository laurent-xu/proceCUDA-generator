#pragma once
#include <app/rendering.hh>

void
render_grids(const Camera& camera, sf::Window* window, bool& running, bool& moved,
             std::shared_ptr<std::shared_ptr<std::vector<rendering::VerticesGrid>>> vertices)
{
  glm::mat4 model;
  glm::mat4 view;
  glm::vec3 lightPos(-4.0f, -13.0f, 9.0f);
  // Note that we're translating the scene in the reverse direction of
  // where we want to move
  view = glm::translate(view, glm::vec3(0.0f, 0.0f, 0.0f));
  glm::mat4 projection;
  projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f);
  Shader shader("../resources/shaders/vertex_shader.glsl",
                "../resources/shaders/fragment_shader.glsl");
  sf::Vector2f mousePosition(sf::Mouse::getPosition());
  std::shared_ptr<>
  while (running)
  {
    // Check if window is closed
    sf::Event event;
    while (window->pollEvent(event))
      if (event.type == sf::Event::Closed)
        running = false;


    // Get input and update camera position
    moved = updateCamera(camera, clock.getElapsedTime().asSeconds(),
                         mousePosition);
    clock.restart();

    // Clean the frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
  if (!vertices && !move)
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [] {return moved || vertices;});
  }
}
