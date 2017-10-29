#include "AST.hh"

class Visitor
{
public:
  virtual void visit(AdditionNode& n) = 0;
  virtual void visit(ClampNode& n) = 0;
  virtual void visit(ConstantNode& n) = 0;
  virtual void visit(MultiplyNode& n) = 0;
  virtual void visit(PerlinNode& n) = 0;
  virtual void visit(PolynomNode& n) = 0;
  virtual void visit(SpatializeCubeMapNode& n) = 0;
  virtual void visit(SpatializeNode& n) = 0;
  virtual void visit(SphereNode& n) = 0;
};
