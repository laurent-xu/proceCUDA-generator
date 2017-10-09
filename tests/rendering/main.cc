#include <iostream>
#include <vector>
#include <rendering/dual-contouring.hh>

int main() {
  std::vector<std::vector<rendering::node_t>> nodes(1);
  for (int z = 0; z < 1; z++) {
    for (int y = 0; y < 5; y++) {
      for (int x = 0; x < 5; x++) {
        int r = rand() % 3;
        if (r == 0)
          nodes[z].push_back(rendering::node_t(-1, rendering::point_t(0, 0, 0)));
        else
          nodes[z].push_back(rendering::node_t(+1, rendering::point_t(0, 0, 0)));
      }
    }
  }
  rendering::HermitianGrid g(nodes, rendering::point_t(5, 5, 1), 1);
  for (int z = 0; z < 1; z++) {
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
  }
  return 0;
}