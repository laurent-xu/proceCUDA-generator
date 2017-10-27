#include <iostream>
#include "density/Sphere.hh"

int main()
{
  auto grid = make_sphere_example({0., 0., 0.}, 1., {16., 16., 16}, 5.);
  return 0;
}
