//logger.h
#pragma once

#include "NvInferRuntimeCommon.h"
#include <iostream>


class logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;
