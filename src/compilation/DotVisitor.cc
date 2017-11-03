#include <compilation/DotVisitor.hh>

namespace
{
  std::string instantiate_node(Node& n,
                               const std::vector<std::string>& params = {})
  {
    auto result = n.output_name + "[shape=rect "
                                  "label=<<B>" + n.output_name + "</B><br/>";
    for (auto& param: params)
      result += "<br/>" + param;
    result += ">];\n";
    return result;
  }
}

std::string DotVisitor::get_str() const
{
  return "digraph g {\n" + str + "\n}\n";
}

void DotVisitor::visit(AdditionNode& n)
{
  str += instantiate_node(n);
  for (auto& child: n.inputs)
  {
    child->accept(*this);
    str += child->output_name + "->" + n.output_name + ";\n";
  }
}

void DotVisitor::visit(ClampNode& n)
{
  auto params = std::vector<std::string>();
  params.push_back("min " + n.min);
  params.push_back("max " + n.max);
  str += instantiate_node(n, params);
  n.input->accept(*this);
  str += n.input->output_name + "->" + n.output_name + ";\n";
}

void DotVisitor::visit(ConstantNode& n)
{
  auto params = std::vector<std::string>();
  params.push_back("value " + n.val);
  str += instantiate_node(n, params);
}

void DotVisitor::visit(MultiplyNode& n)
{
  str += instantiate_node(n);
  for (auto& child: n.inputs)
  {
    child->accept(*this);
    str += child->output_name + "->" + n.output_name + ";\n";
  }
}

void DotVisitor::visit(PerlinNode& n)
{
  auto params = std::vector<std::string>();
  params.push_back("normalization " + n.normalization);
  params.push_back("seed " + n.seed);
  str += instantiate_node(n, params);
}

void DotVisitor::visit(PolynomNode& n)
{
  auto params = std::vector<std::string>();
  for (size_t i = 0; i < n.coef.size(); ++i)
  {
    std::string line = n.coef[i] + " * X^" + std::to_string(i);
    if (i != n.coef.size() - 1)
      line += " +";
    params.push_back(line);
  }
  str += instantiate_node(n, params);
  n.input->accept(*this);
  str += n.input->output_name + "->" + n.output_name + ";\n";
}

void DotVisitor::visit(SpatializeCubeMapNode& n)
{
  auto params = std::vector<std::string>();
  params.push_back("center_x " + n.center_x);
  params.push_back("center_y " + n.center_y);
  params.push_back("center_z " + n.center_z);
  params.push_back("radius " + n.radius);
  params.push_back("min_radius " + n.min_radius);
  params.push_back("max_radius " + n.max_radius);
  str += instantiate_node(n, params);
  n.input->accept(*this);
  str += n.input->output_name + "->" + n.output_name + ";\n";
}

void DotVisitor::visit(SpatializeNode& n)
{
  auto params = std::vector<std::string>();
  params.push_back("center_x " + n.center_x);
  params.push_back("center_y " + n.center_y);
  params.push_back("center_z " + n.center_z);
  params.push_back("radius " + n.radius);
  params.push_back("min_radius " + n.min_radius);
  params.push_back("max_radius " + n.max_radius);
  str += instantiate_node(n, params);
  n.input->accept(*this);
  str += n.input->output_name + "->" + n.output_name + ";\n";
}

void DotVisitor::visit(SphereNode& n)
{
  auto params = std::vector<std::string>();
  params.push_back("center_x " + n.center_x);
  params.push_back("center_y " + n.center_y);
  params.push_back("center_z " + n.center_z);
  params.push_back("radius " + n.radius);
  str += instantiate_node(n, params);
}
