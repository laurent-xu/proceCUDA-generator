#pragma once
#include <compilation/Visitor.hh>
#include <string>

struct Function
{
  std::string function_name;
  std::string output;
  std::vector<std::string> lines;
};

class CompileVisitor: public Visitor
{
  public:
    CompileVisitor(const std::string output):
      functions({Function{"kernel_f3", output, {}}}) {}
    std::string get_cc() const;
    std::string get_cu() const;

    virtual void visit(AdditionNode& n);
    virtual void visit(ClampNode& n);
    virtual void visit(ConstantNode& n);
    virtual void visit(MultiplyNode& n);
    virtual void visit(PerlinNode& n);
    virtual void visit(PolynomNode& n);
    virtual void visit(SpatializeCubeMapNode& n);
    virtual void visit(SpatializeNode& n);
    virtual void visit(SphereNode& n);
  private:
    std::vector<Function> functions;
    size_t current_function = 0;
};
