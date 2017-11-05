#include <compilation/CheckVisitor.hh>

void CheckVisitor::visit(AdditionNode& n)
{
  for (auto& child: n.inputs)
    child->accept(*this);
}

void CheckVisitor::visit(ClampNode& n)
{
  n.input->accept(*this);
}

void CheckVisitor::visit(ConstantNode&)
{
}

void CheckVisitor::visit(MultiplyNode& n)
{
  for (auto& child: n.inputs)
    child->accept(*this);
}

void CheckVisitor::visit(PerlinNode&)
{
}

void CheckVisitor::visit(PolynomNode& n)
{
  n.input->accept(*this);
}

void CheckVisitor::visit(SpatializeCubeMapNode& n)
{
  ++count_spatialize;
  n.input->accept(*this);
}

void CheckVisitor::visit(SpatializeNode& n)
{
  ++count_spatialize;
  n.input->accept(*this);
}

void CheckVisitor::visit(SphereNode&)
{
}
