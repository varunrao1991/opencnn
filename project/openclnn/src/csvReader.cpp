#include "csvReader.h"
#include "logger.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

typedef std::vector<float> record_t;
typedef std::vector<record_t> data_t;

std::istream &operator>>(std::istream &ins, record_t &record)
{
    record.clear();

    std::string line;
    getline(ins, line);

    std::stringstream ss(line);
    std::string field;
    while (getline(ss, field, ','))
    {
        std::stringstream stringstream(field);
        float f = 0.0;
        stringstream >> f;

        record.push_back(f);
    }

    return ins;
}

std::istream &operator>>(std::istream &ins, data_t &data)
{
    data.clear();

    record_t record;
    while (ins >> record)
    {
        if (!record.empty())
            data.push_back(record);
    }

    return ins;
}

CsvReader::CsvReader(float *dataBuffer, std::string filename)
{
    data_t data;

    std::ifstream infile(filename);
    infile >> data;

    if (!infile.eof())
    {
        ALOG_GPUML("Fooey! Complain if something went wrong");
        return;
    }

    infile.close();

    unsigned max_record_size = 0;
    for (unsigned n = 0; n < data.size(); n++)
        if (max_record_size < data[n].size())
            max_record_size = data[n].size();

    for (int j = 0; j < data.size() * max_record_size; j++)
        dataBuffer[j] = data[j / max_record_size][j % max_record_size];
}

CsvReader::~CsvReader()
{
    ALOG_GPUML("Deallocated");
}
