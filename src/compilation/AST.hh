#pragma once
#include <string>
#include <iostream>
#include <density/F3Grid.hh>
#include "parser.hh"

class Visitor;

class Node
{
public:
  using astnode_t = std::shared_ptr<Node>;
  Node(const std::string& type): output_name(type + std::to_string(id++)) {}
  ~Node() = default;
  virtual void accept(Visitor& v) = 0;
  static astnode_t parse(const std::string& str);
  static astnode_t parse(ParsingStringStream& parsing_stream);

  std::string output_name;
  static size_t id;
};

class AdditionNode: public Node
{
public:
  using astnode_t = Node::astnode_t;
  AdditionNode(const std::vector<astnode_t>& inputs)
    : Node("add"),
      inputs(inputs)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const std::vector<astnode_t> inputs;
};

class ClampNode: public Node
{
public:
  using astnode_t = Node::astnode_t;
  ClampNode(const std::string& min,
            const std::string& max,
            const astnode_t& input)
    : Node("clamp"),
      min(min),
      max(max),
      input(input)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const astnode_t input;
  const std::string min;
  const std::string max;
};

class ConstantNode: public Node
{
public:
  ConstantNode(const std::string& val)
    : Node("constant"),
      val(val)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const std::string val;
};

class MultiplyNode: public Node
{
public:
  using astnode_t = Node::astnode_t;
  MultiplyNode(const std::vector<astnode_t>& inputs)
    : Node("multiply"),
      inputs(inputs)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const std::vector<astnode_t> inputs;
};

class PerlinNode: public Node
{
public:
  PerlinNode(const std::string& normalization, const std::string& seed)
    : Node("perlin"),
      normalization(normalization),
      seed(seed)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const std::string normalization;
  const std::string seed;
};

class PolynomNode: public Node
{
public:
  using astnode_t = Node::astnode_t;
  PolynomNode(const astnode_t& input, const std::vector<std::string>& coef)
    : Node("polynom"),
      input(input),
      coef(coef)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const astnode_t input;
  const std::vector<std::string> coef;
};

class SpatializeCubeMapNode: public Node
{
public:
  using astnode_t = Node::astnode_t;
  SpatializeCubeMapNode(const std::string& center_x,
                        const std::string& center_y,
                        const std::string& center_z,
                        const std::string& radius,
                        const std::string& min_radius,
                        const std::string& max_radius,
                        const astnode_t& input)
    : Node("spatializecubemap"),
      center_x(center_x),
      center_y(center_y),
      center_z(center_z),
      radius(radius),
      min_radius(min_radius),
      max_radius(max_radius),
      input(input)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const std::string center_x;
  const std::string center_y;
  const std::string center_z;
  const std::string radius;
  const std::string min_radius;
  const std::string max_radius;
  const astnode_t input;
};

class SpatializeNode: public Node
{
public:
  using astnode_t = Node::astnode_t;
  SpatializeNode(const std::string& center_x,
                 const std::string& center_y,
                 const std::string& center_z,
                 const std::string& radius,
                 const std::string& min_radius,
                 const std::string& max_radius,
                 const astnode_t& input)
    : Node("spatialize"),
      center_x(center_x),
      center_y(center_y),
      center_z(center_z),
      radius(radius),
      min_radius(min_radius),
      max_radius(max_radius),
      input(input)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const std::string center_x;
  const std::string center_y;
  const std::string center_z;
  const std::string radius;
  const std::string min_radius;
  const std::string max_radius;
  const astnode_t input;
};

class SphereNode: public Node
{
public:
  SphereNode(const std::string& center_x,
             const std::string& center_y,
             const std::string& center_z,
             const std::string& radius)
  : Node("sphere"),
    center_x(center_x),
    center_y(center_y),
    center_z(center_z),
    radius(radius)
  {
  }
  static astnode_t parse(ParsingStringStream& parsing_stream);

  virtual void accept(Visitor& v) override;

  const std::string center_x;
  const std::string center_y;
  const std::string center_z;
  const std::string radius;
};
