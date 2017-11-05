#pragma once
#include <compilation/Visitor.hh>
#include <string>

class CheckVisitor: public Visitor
{
  public:
    bool is_valid() const { return count_spatialize <= 1; }

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
    size_t count_spatialize = 0;
};
