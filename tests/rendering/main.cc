#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <rendering/hermitian-grid.hh>
#include <rendering/utils/nm-matrix.hpp>
#include <rendering/qr-decomposition.hpp>
#include <density/Sphere.hh>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <rendering/viewer/camera.hh>
#include <rendering/viewer/shader.hh>
#include <rendering/vertices-grid.hpp>

using data_t = rendering::data_t;
using point_t = rendering::point_t;
using node_t = rendering::node_t;
using nmMatrix = rendering::utils::nmMatrix;

// Camera controls
void updateCamera(Camera &camera, float deltaTime, sf::Vector2f &formerPosition);

void testCube() {
  // Creating window
  sf::Window *window = new sf::Window(sf::VideoMode(800, 600), "TestSphere",
                                      sf::Style::Default, sf::ContextSettings(32));
  window->setVerticalSyncEnabled(true);
  glEnable(GL_DEPTH_TEST);
  if (glewInit() == GLEW_OK)
    std::cout << "Glew initialized successfully" << std::endl;

  auto sphere = make_sphere_example(F3::vec3_t(0, 0, 0), F3::dist_t(1), F3::vec3_t(16, 16, 16), F3::dist_t(10));
  auto dimension = sphere->dim_size();
  rendering::HermitianGrid
      hermitianGrid(sphere, point_t(dimension, dimension, dimension), 1);
  rendering::VerticesGrid verticesGrid(hermitianGrid, 1);
  verticesGrid.initVAO();

  std::vector<GLfloat> v_int;
  auto &grid = hermitianGrid.getGrid();
  auto dim = hermitianGrid.getDimensions();
  for (int k = 0; k < dim.z; k++)
    for (int y = 0; y < dim.y; y++)
      for (int x = 0; x < dim.x; x++) {
        auto &node = grid[k][y * dim.x + x];
        if (node.intersections.x != node.min.x
            || node.intersections.y != node.min.y
            || node.intersections.z != node.min.z)
        {
          v_int.push_back(node.intersections.x);
          v_int.push_back(node.intersections.y);
          v_int.push_back(node.intersections.z);
        }
      }
  GLfloat *ints = &(v_int[0]);
  auto ints_size = static_cast<GLuint>(v_int.size());
  GLuint VAO;
  GLuint VBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO); {
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, ints_size * sizeof (GLfloat), ints, GL_STATIC_DRAW);
    // vertices
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid *) (0));
    glEnableVertexAttribArray(0);
  } glBindVertexArray(0);

  glm::mat4 model;
  glm::mat4 view;
  glm::vec3 lightPos(-4.0f, -13.0f, 9.0f);
  Camera camera;
  // Note that we're translating the scene in the reverse direction of where we want to move
  view = glm::translate(view, glm::vec3(0.0f, 0.0f, 0.0f));
  glm::mat4 projection;

  projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f);
  Shader shader("../resources/shaders/vertex_shader.glsl",
                "../resources/shaders/fragment_shader.glsl");

  glPointSize(10);

  glClearColor(0.1, 0.1, 0.1, 1.0);
  bool running = true;
  sf::Clock clock;
  sf::Clock rotationClock;
  sf::Vector2f mousePosition(sf::Mouse::getPosition());
  glm::vec3 lightPosBase = lightPos;
  while (running) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    sf::Event event;
    while (window->pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        running = false;
      else if (event.type == sf::Event::Resized) {
        glViewport(0, 0, event.size.width, event.size.height);
        projection = glm::perspective(glm::radians(45.0f),
                                      (float) event.size.width / (float) event.size.height, 0.1f, 1000.0f);
      }
    }
    updateCamera(camera, clock.getElapsedTime().asSeconds(), mousePosition);
    clock.restart();
    shader.Use();

    GLint pointLightPosLoc = glGetUniformLocation(shader.getProgram(), "pointLightPos");
    GLint pointLightColorLoc  = glGetUniformLocation(shader.getProgram(), "pointLightColor");
    glUniform3f(pointLightColorLoc,  1.0f, 1.0f, 1.0f); // Also set light's color (white)
    glUniform3f(pointLightPosLoc, camera.getPosition().x, camera.getPosition().y,
                camera.getPosition().z);

    GLint lightDir1Loc = glGetUniformLocation(shader.getProgram(), "lightDir1");
    GLint objectColorLoc = glGetUniformLocation(shader.getProgram(), "objectColor");
    GLint lightColor1Loc  = glGetUniformLocation(shader.getProgram(), "lightColor1");
    glUniform3f(objectColorLoc, 1.0f, 0.5f, 0.31f);
    glUniform3f(lightColor1Loc,  1.0f, 1.0f, 6.0f); // Also set light's color (white)
    glUniform3f(lightDir1Loc, 1.0f, 1.0f, 1.0f);

    GLint modelLoc = glGetUniformLocation(shader.getProgram(), "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    GLint viewLoc = glGetUniformLocation(shader.getProgram(), "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(camera.getViewMatrix()));
    GLint projLoc = glGetUniformLocation(shader.getProgram(), "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    GLint viewerLocPos = glGetUniformLocation(shader.getProgram(), "viewerPos");
    glUniform3f(viewerLocPos, camera.getPosition().x, camera.getPosition().y,
                camera.getPosition().z);

    verticesGrid.draw();
    /*
    glBindVertexArray(VAO); {
      glDrawArrays(GL_POINTS, 0, (GLsizei) ints_size);
    } glBindVertexArray(0);
    */
    window->display();
  }
  delete window;
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

void testMatrixNM() {
  using nmMatrix = rendering::utils::nmMatrix;
  int n1 = 3, m1 = 4;
  int n2 = 4, m2 = 3;
  int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  int b[] = { 4, 7, 3, 6, 9, 1, 11, 12, 10, 2, 8, 5 };
  std::vector<float> A;
  std::vector<float> B;
  A.assign(a, a + n1 * m1);
  B.assign(b, b + n2 * m2);
  std::cout << "Matrix A:" << std::endl;
  nmMatrix::print(A, n1, m1);
  std::cout << "Matrix B:" << std::endl;
  nmMatrix::print(B, n2, m2);
  std::cout << "Matrix C = A * B:" << std::endl;
  auto C = nmMatrix::multiply(A, B, n1, m1, n2, m2);
  nmMatrix::print(C, n1, m2);
  std::cout << "Matrix A^t:" << std::endl;
  auto At = nmMatrix::transpose(A, n1, m1);
  nmMatrix::print(At, m1, n1);
  std::cout << "Matrix D: A^t + B" << std::endl;
  auto D = nmMatrix::add(At, B, n2, m2);
  nmMatrix::print(D, n2, m2);
  std::cout << "Matrix E: D::B" << std::endl;
  auto E = nmMatrix::append(D, B, n2, m2, m2);
  nmMatrix::print(E, n2, m2 + m2);
  std::cout << "Matrix F: extract E" << std::endl;
  auto F = nmMatrix::extract(E, 2, 1, 4, 4, 6);
  nmMatrix::print(F, 2, 2);
}

void testQRDecomposition() {
  float a1[] = { 6, 5, 0, 5, 1, 4, 0, 4, 3 };
  std::vector<data_t> m1;
  m1.assign(a1, a1 + 9);
  std::cout << "Start matrix:" << std::endl;
  nmMatrix::print(m1, 3, 3, 12);
  std::cout << std::endl;
  rendering::QRDecomposition qrd1(m1, 3, 3);
  std::cout << "Processed matrix:" << std::endl;
  nmMatrix::print(qrd1.getProcessedMatrix(), 3, 3, 12);
  std::cout << std::endl;
  std::cout << "A^:" << std::endl;
  nmMatrix::print(qrd1.extractAa(), 2, 2, 12);
  std::cout << std::endl;
  std::cout << "B^:" << std::endl;
  nmMatrix::print(qrd1.extractBb(), 2, 1, 12);
  std::cout << std::endl;
  std::cout << "r: " << qrd1.getR() << std::endl;

  float a2[] = { -1, 9, 2, 8, 7, 5, 6, -5, 7, 2, -9, 0, 1, 2, -3 };
  std::vector<data_t> m2;
  m2.assign(a2, a2 + 15);
  std::cout << "Start matrix:" << std::endl;
  nmMatrix::print(m2, 3, 5, 12);
  std::cout << std::endl;
  rendering::QRDecomposition qrd2(m2, 3, 5);
  std::cout << "Processed matrix:" << std::endl;
  nmMatrix::print(qrd2.getProcessedMatrix(), 3, 5, 12);
  std::cout << std::endl;
  std::cout << "A^:" << std::endl;
  nmMatrix::print(qrd2.extractAa(), 3, 4, 12);
  std::cout << std::endl;
  std::cout << "b^:" << std::endl;
  nmMatrix::print(qrd2.extractBb(), 3, 1, 12);
  std::cout << std::endl;
}

void testHermiteanComputation() {
  std::vector<std::vector<rendering::node_t>> nodes(2);
  for (int z = 0; z < 2; z++)
    for (int y = 0; y < 5; y++)
      for (int x = 0; x < 5; x++) {
        int r = rand() % 3;
        if (r == 0)
          nodes[z].push_back(rendering::node_t(-1, rendering::point_t(0, 0, 0)));
        else
          nodes[z].push_back(rendering::node_t(+1, rendering::point_t(0, 0, 0)));
      }
  rendering::HermitianGrid::printDensityGrid(nodes, rendering::point_t(5, 5, 2));
  rendering::HermitianGrid g(nodes, rendering::point_t(5, 5, 2), 1);
  for (int z = 0; z < 2; z++)
    for (int y = 0; y < 5; y++) {
      for (int x = 0; x < 5; x++) {
        auto e = nodes[z][y * 5 + x].value;
        if (e == -1)
          std::cout << ". ";
        if (e == 0)
          std::cout << "O ";
        if (e == 1)
          std::cout << "0 ";
      }
      std::cout << std::endl;
    }
  std::cout << std::endl;
  for (int z = 0; z < 2; z++)
    for (int y = 0; y < 5; y++) {
      for (int x = 0; x < 5; x++) {
        auto e = g.getValueAt(x, y, z).value;
        if (e == -1)
          std::cout << ". ";
        if (e == 0)
          std::cout << "O ";
        if (e == 1)
          std::cout << "0 ";
      }
      std::cout << std::endl;
    }
  rendering::HermitianGrid::printHermitianGrid(g.getGrid(), g.getDimensions());
}

void testDualContouring() {
  auto sphere = make_sphere_example(F3::vec3_t(0, 0, 0), F3::dist_t(1), F3::vec3_t(16, 16, 16), F3::dist_t(10));
  auto dimension = sphere->dim_size();
  rendering::HermitianGrid
          hermitianGrid(sphere, point_t(dimension, dimension, dimension), 1);
  rendering::VerticesGrid verticesGrid(hermitianGrid, 1);
  std::vector<GLfloat> vertices;
  // hermitianGrid.printHermitianGrid(hermitianGrid.getGrid(), hermitianGrid.getDimensions());
}

int main() {
  //testMatrixNM();
  //testQRDecomposition();
  //testHermiteanComputation();
  testCube();
  //testDualContouring();
  return 0;
}
