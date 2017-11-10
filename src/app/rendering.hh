#pragma once
#include <rendering/hermitian-grid.hh>
#include <rendering/utils/nm-matrix.hpp>
#include <rendering/qr-decomposition.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include <rendering/viewer/camera.hh>
#include <rendering/viewer/shader.hh>
#include <rendering/vertices-grid.hpp>
#include <vector>

void
render_grids(const Camera& camera, sf::Window* window, bool& running
             std::shared_ptr<std::vector<rendering::VerticesGrid>>* vertices);
