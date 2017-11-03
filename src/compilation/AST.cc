#include "AST.hh"
#include "Visitor.hh"

size_t Node::id = 0;

void AdditionNode::accept(Visitor& v)
{
  v.visit(*this);
}

void ClampNode::accept(Visitor& v)
{
  v.visit(*this);
}

void ConstantNode::accept(Visitor& v)
{
  v.visit(*this);
}

void MultiplyNode::accept(Visitor& v)
{
  v.visit(*this);
}

void PerlinNode::accept(Visitor& v)
{
  v.visit(*this);
}

void PolynomNode::accept(Visitor& v)
{
  v.visit(*this);
}

void SpatializeCubeMapNode::accept(Visitor& v)
{
  v.visit(*this);
}

void SpatializeNode::accept(Visitor& v)
{
  v.visit(*this);
}
void SphereNode::accept(Visitor& v)
{
  v.visit(*this);
}
