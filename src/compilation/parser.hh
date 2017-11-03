#pragma once
#include <string>
#include <vector>

enum class Token
{
  ADDITION,
  CLAMP,
  CONSTANT,
  MULTIPLY,
  PERLIN,
  POLYNOM,
  SPATIALIZECUBE,
  SPATIALIZE,
  SPHERE,
  OPEN,
  CLOSE,
  NUMBER,
  END,
  NONE
};

struct Lexeme
{
  Lexeme(Token type, const std::string& str): type(type), str(str) {}

  Token type;
  std::string str;
};

class ParsingStringStream
{
public:
  ParsingStringStream(std::string s);

  bool has_token() { return index < lexemes.size(); }
  Lexeme peak_next_lexeme();
  Lexeme get_next_lexeme();
  Lexeme get_next_lexeme_expected(Token expected, const std::string& caller);

private:
  std::vector<Lexeme> lexemes;
  size_t index = 0;
};

