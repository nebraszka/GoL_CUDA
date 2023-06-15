#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>

void brk() {}

#define CHECK_CUDA(call)                                           \
do                                                                 \
{                                                                  \
    cudaError_t status = call;                                     \
    if (status != cudaSuccess) {                                   \
        std::cerr << "Error at " << __FILE__ << ":" << __LINE__    \
                  << " - " << cudaGetErrorString(status)           \
                  << std::endl;                                    \
        brk();                                                     \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
}                                                                  \
while(false)
