#include <compilation/CompileVisitor.hh>

namespace
{
std::string get_sources(bool is_cuda, const std::vector<Function>& functions)
{
  std::string result = "#include <density/F3Grid.hh>\n"
                       "#include <app/generation_kernel.hh>\n"
                       "#include <density/primitive.hh>\n\n";
  std::string target = is_cuda ? "__device__ " : "";

  for (const auto& f: functions)
    result += target + "F3 " + f.function_name +
              "(const F3::vec3_t& position);\n";

  for (const auto& f: functions)
  {
    result += target + "F3 " + f.function_name +
              "(const F3::vec3_t& position)\n";
    result += "{\n";
    result += "  (void)position;\n";
    for (const auto& l: f.lines)
      result += "  " + l + "\n";
    result += "  return " + f.output + ";\n";
    result += "}\n";
  }

  return result;
}
}

std::string CompileVisitor::get_cc() const
{
  return get_sources(false, functions);
}

std::string CompileVisitor::get_cu() const
{
  auto result = get_sources(true, functions);
  result += "__global__ void kernel_f3_caller(GridF3<true> grid)\n"
            "{\n"
            "  size_t x = blockDim.x * blockIdx.x + threadIdx.x;\n"
            "  size_t y = blockDim.y * blockIdx.y + threadIdx.y;\n"
            "  size_t z = blockDim.z * blockIdx.z + threadIdx.z;\n"
            "  size_t dimension = grid.dim_size();\n"
            "\n"
            "  if (x < dimension && y < dimension && z < dimension)\n"
            "  {\n"
            "    auto position = grid.to_position(x, y, z);\n"
            "    grid.at(x, y, z) = kernel_f3(position);\n"
            "  }\n"
            "}\n";
  return result;
}

void CompileVisitor::visit(AdditionNode& n)
{
  for (auto& child: n.inputs)
    child->accept(*this);
  auto& f = functions[current_function];
  std::string line = "F3 " + n.output_name + " = ";
  line += "density::add(";
  for (auto& child: n.inputs)
    line += child->output_name + ", ";
  line[line.size() - 2] = ')';
  line[line.size() - 1] = ';';
  f.lines.push_back(line);
}

void CompileVisitor::visit(ClampNode& n)
{
  n.input->accept(*this);
  auto& f = functions[current_function];
  std::string line = "F3 " + n.output_name + " = ";
  line += "density::clamp(" + n.min + ", " + n.max +
          "," + n.input->output_name + ");";
  f.lines.push_back(line);
}

void CompileVisitor::visit(ConstantNode& n)
{
  auto& f = functions[current_function];
  std::string line = "F3 " + n.output_name + " = ";
  line += "{" + n.val + ", F3::vec3_t(0., 0., 0.)};";
  f.lines.push_back(line);
}

void CompileVisitor::visit(MultiplyNode& n)
{
  for (auto& child: n.inputs)
    child->accept(*this);
  auto& f = functions[current_function];
  std::string line = "F3 " + n.output_name + " = ";
  line += "density::multiply(";
  for (auto& child: n.inputs)
    line += child->output_name + ", ";
  line[line.size() - 2] = ')';
  line[line.size() - 1] = ';';
  f.lines.push_back(line);
}

void CompileVisitor::visit(PerlinNode& n)
{
  auto& f = functions[current_function];
  std::string line = "F3 " + n.output_name + " = ";
  line += "density::perlin(position, " + n.normalization +
          ", (size_t)" + n.seed + ");";
  f.lines.push_back(line);
}

void CompileVisitor::visit(PolynomNode& n)
{
  n.input->accept(*this);
  auto& f = functions[current_function];
  std::string line = "  F3 " + n.output_name + " = ";
  line += "density::polynom(" + n.input->output_name;
  for (auto& c: n.coef)
    line += ", " + c;
  line += ");";
  f.lines.push_back(line);

}

void CompileVisitor::visit(SpatializeCubeMapNode& n)
{
  n.input->accept(*this);
  std::cerr << "SpatializeCubeMap is not implemented yet" << std::endl;
  std::exit(1);
}

void CompileVisitor::visit(SpatializeNode& n)
{
  std::string projected_position = "projected_position_" + n.output_name;
  std::string dist = "dist_" + n.output_name;
  std::string center = "center_" + n.output_name;
  std::string height = "height_" + n.output_name;
  std::string height_function = height + "_f";

  functions.push_back(Function{height_function, n.input->output_name, {}});
  size_t old_index = current_function;
  current_function = functions.size() - 1;
  n.input->accept(*this);
  current_function = old_index;

  auto& f = functions[current_function];
  f.lines.push_back("F3 " + n.output_name +
                    " = {0. , F3::vec3_t(0., 0., 0.)};");
  f.lines.push_back("F3::vec3_t " + center + " = F3::vec3_t(" +
                    n.center_x + ", " + n.center_y + ", " + n.center_z + ");");
  f.lines.push_back("F3::dist_t " + dist +
                    " = glm::distance(position, " + center + ");");
  f.lines.push_back("if (" + dist + " > " + n.min_radius +
                    " && " + dist + " < " + n.max_radius + ")");
  f.lines.push_back("{");
  f.lines.push_back("  F3::vec3_t " + projected_position +
                    " = density::project_spatialize(" + center +
                    ", " + n.radius + ", position);");
  f.lines.push_back("  F3 " + height + " = " + height_function + "("
                    + projected_position + ");");
  f.lines.push_back("  " + n.output_name + " = density::spatialize(" + height +
                    ", " + n.radius + ");");
  f.lines.push_back("}");
}

void CompileVisitor::visit(SphereNode& n)
{
  auto& f = functions[current_function];
  std::string line = "F3 " + n.output_name + " = ";
  line += "density::sphere(F3::vec3_t(" +
          n.center_x + ", " + n.center_y + ", " + n.center_z + "), " +
          n.radius + ", position);";
  f.lines.push_back(line);
}
