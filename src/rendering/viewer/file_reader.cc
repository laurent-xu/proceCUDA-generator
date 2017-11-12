//
// Created by leo on 7/31/16.
//

#include <utils/cudamacro.hh>
#include "file_reader.hh"
#include <fstream>
#include <iostream>

std::string file_reader::readFile(std::string fileName) {
  std::ifstream ifs;
  try
  {
    ifs.open(fileName);
    if (!ifs.is_open())
      CERR << "<FileReader> ERROR: Cannot open file "
           << fileName << std::endl;
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        (std::istreambuf_iterator<char>()));
    return content;
  }
  catch (std::ifstream::failure e)
  {
    CERR << "ERROR::FILE_READER::FILE_NOT_SUCCESSFULLY_READ("
         << fileName << ")" << std::endl;
  }
  return "";
}
