
#pragma once

#include <string>

class CsvReader
{
  public:
    CsvReader(float *buffer, std::string filename);
    virtual ~CsvReader();
};
