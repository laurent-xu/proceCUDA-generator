#include <iostream>
#include <fstream>
#include <streambuf>
#include <density/Sphere.hh>
#include <compilation/AST.hh>
#include <compilation/DotVisitor.hh>

int main(int argc, char* argv[])
{
  if (argc == 3)
  {
    auto fs = std::ifstream(argv[1]);
    auto str = std::string(std::istreambuf_iterator<char>(fs),
                           std::istreambuf_iterator<char>());
    auto root = Node::parse(str);

    auto v = DotVisitor();
    root->accept(v);

    auto out_fs = std::ofstream(argv[2]);
    if (out_fs)
    {
      out_fs << v.get_str() << std::endl;
      out_fs.close();
    }
  }
  else
  {
    std::cerr << "help: "<< argv[0] << " input.txt output.dot" << std::endl;
    return 1;
  }
  return 0;
}
