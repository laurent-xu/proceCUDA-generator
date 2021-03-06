#include "parser.hh"
#include "AST.hh"
#include <unordered_map>
#include <regex>

namespace
{
  struct EnumClassHash
  {
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
  };

  std::string token_to_string(Token type)
  {
    std::unordered_map<Token, std::string, EnumClassHash> map =
    {
      {Token::ADDITION, "AdditionToken"},
      {Token::CLAMP, "ClampToken"},
      {Token::CONSTANT, "ConstantToken"},
      {Token::MULTIPLY, "MultiplyToken"},
      {Token::PERLIN, "PerlinToken"},
      {Token::POLYNOM, "PolynomToken"},
      {Token::SPATIALIZECUBE, "SpatializeCubeToken"},
      {Token::SPATIALIZE, "SpatializeToken"},
      {Token::SPHERE, "SphereToken"},
      {Token::OPEN, "OpenToken"},
      {Token::CLOSE, "CloseToken"},
      {Token::NUMBER, "NumberToken"},
      {Token::END, "EndToken"},
      {Token::NONE, "NoneToken"}
    };
    return map[type];
  }

  std::string token_to_string(Lexeme lex)
  {
    return token_to_string(lex.type) + "\'" + lex.str + "\'";
  }
}

ParsingStringStream::ParsingStringStream(std::string str)
{
  std::vector<std::pair<Token, std::string>> map =
  {
    {Token::ADDITION, R"(\s*(add))"},
    {Token::CLAMP, R"(\s*(clamp))"},
    {Token::CONSTANT, R"(\s*(constant))"},
    {Token::MULTIPLY, R"(\s*(multiply))"},
    {Token::PERLIN, R"(\s*(perlin))"},
    {Token::POLYNOM, R"(\s*(polynom))"},
    {Token::SPATIALIZECUBE, R"(\s*(spatializecubemap))"},
    {Token::SPATIALIZE, R"(\s*(spatialize))"},
    {Token::SPHERE, R"(\s*(sphere))"},
    {Token::OPEN, R"(\s*(\())"},
    {Token::CLOSE, R"(\s*(\)))"},
    {Token::NUMBER, R"(\s*([-+]?\d*(\d+|(\.\d*))(?:[eE]([-+]?\d+))?))"},
    {Token::END, R"(\s+)"}
  };

  while (!str.empty())
  {
    bool found = false;
    for (const auto& p: map)
    {
      auto e = std::regex(p.second, std::regex_constants::icase);
      auto m = std::smatch();
      if (std::regex_search(str, m, e, std::regex_constants::match_continuous))
      {
        auto lex = std::string();
        if (p.first != Token::END)
          lex = m[1];
        lexemes.emplace_back(p.first, lex);
        str = m.suffix().str();
        found = true;
        break;
      }
    }
    if (!found)
    {
      std::cerr << str << "doesn't match any token" << std::endl;
      std::exit(1);
    }
  }
}

Lexeme ParsingStringStream::peak_next_lexeme()
{
  if (index >= lexemes.size())
  {
    std::cerr << "No remaining tokens." << std::endl;
    return Lexeme(Token::NONE, "");
  }
  return lexemes[index];
}

Lexeme ParsingStringStream::get_next_lexeme()
{
  auto result = peak_next_lexeme();
  ++index;
  return result;
}

Lexeme ParsingStringStream::get_next_lexeme_expected(Token expected,
                                                     const std::string& caller)
{
  auto result = get_next_lexeme();
  if (expected != result.type)
  {
    std::cerr << caller << ": Token " << token_to_string(result)
              << " is unexpected. " << token_to_string(expected)
              << " is expected instead." << std::endl;
    std::exit(1);
  }
  return result;
}

auto Node::parse(const std::string& str) -> astnode_t
{
  auto stream = ParsingStringStream(str);
  auto result = parse(stream);
  if (stream.has_token())
    stream.get_next_lexeme_expected(Token::END, "function parser");
  return result;
}

auto Node::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  auto func = parsing_stream.get_next_lexeme();
  auto opening = parsing_stream.get_next_lexeme_expected(Token::OPEN,
                                                         "function parser");
  astnode_t result;
  switch (func.type)
  {
  case Token::ADDITION:
    result = AdditionNode::parse(parsing_stream);
    break;
  case Token::CLAMP:
    result = ClampNode::parse(parsing_stream);
    break;
  case Token::CONSTANT:
    result = ConstantNode::parse(parsing_stream);
    break;
  case Token::MULTIPLY:
    result = MultiplyNode::parse(parsing_stream);
    break;
  case Token::PERLIN:
    result = PerlinNode::parse(parsing_stream);
    break;
  case Token::POLYNOM:
    result = PolynomNode::parse(parsing_stream);
    break;
  case Token::SPATIALIZECUBE:
    result = SpatializeCubeMapNode::parse(parsing_stream);
    break;
  case Token::SPATIALIZE:
    result = SpatializeNode::parse(parsing_stream);
    break;
  case Token::SPHERE:
    result = SphereNode::parse(parsing_stream);
    break;
  default:
    std::cerr << "Token " << token_to_string(func) << " is unexpected. "
              << "A function token is expected instead." << std::endl;
    std::exit(1);
    break;
  }
  return result;
}

auto AdditionNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  std::vector<astnode_t> inputs;
  while (parsing_stream.peak_next_lexeme().type != Token::CLOSE)
    inputs.push_back(Node::parse(parsing_stream));
  parsing_stream.get_next_lexeme_expected(Token::CLOSE, "Addition");
  if (inputs.empty())
  {
    std::cerr << "Add must have at least one argument" << std::endl;
    std::exit(1);
  }
  return std::make_shared<AdditionNode>(inputs);
}

auto ClampNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  auto min_value = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                           "Clamp");
  auto max_value = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                           "Clamp");
  auto input = Node::parse(parsing_stream);
  parsing_stream.get_next_lexeme_expected(Token::CLOSE,
                                          "Clamp");
return std::make_shared<ClampNode>(min_value.str, max_value.str, input);
}

auto ConstantNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  auto value = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                       "Constant");
  parsing_stream.get_next_lexeme_expected(Token::CLOSE,
                                          "Constant");
  return std::make_shared<ConstantNode>(value.str);
}

auto MultiplyNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  std::vector<astnode_t> inputs;
  while (parsing_stream.peak_next_lexeme().type != Token::CLOSE)
    inputs.push_back(Node::parse(parsing_stream));
  parsing_stream.get_next_lexeme_expected(Token::CLOSE,
                                          "Multiply");
  if (inputs.empty())
  {
    std::cerr << "Multiply must have at least one argument" << std::endl;
    std::exit(1);
  }
  return std::make_shared<MultiplyNode>(inputs);
}

auto PerlinNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  auto normalization = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                               "Perlin");
  auto seed = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                      "Perlin");
  parsing_stream.get_next_lexeme_expected(Token::CLOSE,
                                          "Perlin");
  return std::make_shared<PerlinNode>(normalization.str, seed.str);
}

auto PolynomNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  std::vector<std::string> coef;
  auto input = Node::parse(parsing_stream);
  while (parsing_stream.peak_next_lexeme().type != Token::CLOSE)
  {
    auto num = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                       "Polynom");
    coef.push_back(num.str);
  }
  parsing_stream.get_next_lexeme_expected(Token::CLOSE,
                                          "Polynom");
  if (coef.empty())
  {
    std::cerr << "Polynom must have at least one coefficient" << std::endl;
    std::exit(1);
  }
  return std::make_shared<PolynomNode>(input, coef);
}

auto SpatializeCubeMapNode::parse(ParsingStringStream& parsing_stream)
  -> astnode_t
{
  auto center_x = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "SpatializeCubeMap");
  auto center_y = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "SpatializeCubeMap");
  auto center_z = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "SpatializeCubeMap");
  auto radius = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                        "SpatializeCubeMap");
  auto min_radius = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                           "SpatializeCubeMap");
  auto max_radius = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                           "SpatializeCubeMap");
  auto input = Node::parse(parsing_stream);
  parsing_stream.get_next_lexeme_expected(Token::CLOSE,
                                          "SpatializeCubeMap");
  return std::make_shared<SpatializeCubeMapNode>(center_x.str,
                                                 center_y.str,
                                                 center_z.str,
                                                 radius.str,
                                                 min_radius.str,
                                                 max_radius.str,
                                                 input);
}

auto SpatializeNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  auto center_x = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "Spatialize");
  auto center_y = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "Spatialize");
  auto center_z = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "Spatialize");
  auto radius = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                        "Spatialize");
  auto min_radius = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                            "Spatialize");
  auto max_radius = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                            "Spatialize");
  auto input = Node::parse(parsing_stream);
  parsing_stream.get_next_lexeme_expected(Token::CLOSE, "Spatialize");
  return std::make_shared<SpatializeNode>(center_x.str,
                                          center_y.str,
                                          center_z.str,
                                          radius.str,
                                          min_radius.str,
                                          max_radius.str,
                                          input);
}

auto SphereNode::parse(ParsingStringStream& parsing_stream) -> astnode_t
{
  auto center_x = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "Sphere");
  auto center_y = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "Sphere");
  auto center_z = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                          "Sphere");
  auto radius = parsing_stream.get_next_lexeme_expected(Token::NUMBER,
                                                        "Sphere");
  auto max = parsing_stream.get_next_lexeme_expected(Token::CLOSE,
                                                     "Sphere");
  return std::make_shared<SphereNode>(center_x.str,
                                      center_y.str,
                                      center_z.str,
                                      radius.str);
}
