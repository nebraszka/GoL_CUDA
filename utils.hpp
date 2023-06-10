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


std::vector<float> generateCheckerboardTexture(int width, int height, int squareSize)
{
    std::vector<float> texture;
    texture.reserve(width * height * 3); // 3 channels (RGB)

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float r = ((x / squareSize) % 2 == 0) ? ((y / squareSize) % 2 == 0 ? 1.0f : 0.0f) : ((y / squareSize) % 2 == 0 ? 0.0f : 1.0f);
            float g = r;
            float b = r;

            // if (squareSize * 5 <= x && x < squareSize * 6) {
            //     if (squareSize * 5 <= y && y < squareSize * 6) {
            //         g = 0;
            //     }
            // }

            texture.push_back(r);
            texture.push_back(g);
            texture.push_back(b);
            texture.push_back(1.0f);
        }
    }

    return texture;
}