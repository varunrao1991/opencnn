
#pragma once
#include "cl/cl.h"
#include <stdio.h>

#include <chrono>
#include <stdio.h>
#include <string>

//#define ENABLE_MYAPP_LOG_GPUML
//#define ENABLE_INC_TIMER_GPUML

void log(char *format, ...);
void logNoNewline(char *format, ...);
void checkErr(cl_int err, int line, const char *n, bool verbosity = false);
void checkError(cl_int clErrNum, cl_int clSuccess);
const char *getErrorString(cl_int error);

#define LOG_TAG_GPUML "ProjectLeap"

#ifdef ENABLE_INC_TIMER_GPUML

#define INIT_INC_TIMER_GPUML(idx) static std::chrono::milliseconds duration##idx = {};
#define START_INC_TIMER_GPUML(idx) auto start##idx = std::chrono::high_resolution_clock::now()

#define STOP_INC_TIMER_GPUML(idx)                                                            \
    auto stop##idx = std::chrono::high_resolution_clock::now();                              \
    auto elapsedtime = stop##idx - start##idx;                                               \
    auto durationMilli = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedtime); \
    duration##idx += durationMilli;
#define PRINT_INC_TIMER_GPUML(X, idx)                                                 \
    std::cout << X << ": Time taken: " << duration##idx.count() << " milliseconds [ " \
              << float(duration##idx.count()) / 1000 << " seconds ]" << std::endl;

#else
#define START_INC_TIMER_GPUML(...)
#define STOP_INC_TIMER_GPUML(...)
#endif

#ifdef ENABLE_MYAPP_LOG_GPUML
#define ALOG_GPUML(...)           \
    {                             \
        std::printf(__VA_ARGS__); \
        std::printf("\n");        \
    }
#define ALOG_GPUML_MEASURE(...)   \
    {                             \
        std::printf(__VA_ARGS__); \
        std::printf("\n");        \
    }
#define ALOG_GPUML_NO_NEWLINE(...) std::printf(__VA_ARGS__);
#define ALOG_GPUML_MEASURE_NO_NEWLINE(...) std::printf(__VA_ARGS__);
#define ALOG_GPUML_CHECK_ERROR1(err, line, n) checkErr(err, line, n)
#define ALOG_GPUML_CHECK_ERROR2(clErrNum, clSuccess) checkError(clErrNum, clSuccess)
#define GPUML_PRINT_DEVICE_INFO(device) displayDeviceInfo(device)
#else
#define ALOG_GPUML(...)
#define ALOG_GPUML_MEASURE(...)
#define ALOG_GPUML_NO_NEWLINE(...)
#define ALOG_GPUML_MEASURE_NO_NEWLINE(...)
#define ALOG_GPUML_CHECK_ERROR1(err, line, n)
#define ALOG_GPUML_CHECK_ERROR2(clErrNum, clSuccess)
#define displayDeviceInfo(device)
#endif
