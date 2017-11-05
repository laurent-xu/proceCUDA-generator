#include <iostream>
#include <fstream>
#include <streambuf>
#include <density/Sphere.hh>
#include <compilation/AST.hh>
#include <compilation/DotVisitor.hh>

int main(int argc, char* argv[])
{
  if (argc == 4)
  {
    auto fs = std::ifstream(argv[1]);
    auto str = std::string(std::istreambuf_iterator<char>(fs),
                           std::istreambuf_iterator<char>());
    auto root = Node::parse(str);

    /*
    auto v = CompileVisitor();
    root->accept(v);

    auto cu_fs = std::ofstream(argv[2]);
    if (cu_fs)
    {
      cu_fs << v.get_cu() << std::endl;
      cu_fs.close();
    }

    auto cc_fs = std::ofstream(argv[3]);
    if (cc_fs)
    {
      cc_fs << v.get_cc() << std::endl;
      cc_fs.close();
    }*/
    (void)root;
  }
  else
  {
    std::cerr << "usage: "<< argv[0] << " input.proc output.cu output.cc" 
              << std::endl;
    return 1;
  }
  return 0;
}
