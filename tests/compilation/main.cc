#include <iostream>
#include <fstream>
#include <streambuf>
#include <density/Sphere.hh>
#include <compilation/AST.hh>

int main(int argc, char* argv[])
{
  if (argc == 2)
  {
    auto fs = std::ifstream(argv[1]);
    auto str = std::string(std::istreambuf_iterator<char>(fs),
                           std::istreambuf_iterator<char>());
    auto root = Node::parse(str);
  }
  else
  {
    std::cerr << "help: "<< argv[0] << " input.txt" << std::endl;
    return 1;
  }
  return 0;
}
