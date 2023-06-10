#pragma once

#include <cuda_runtime.h>
#include <utils.hpp>


struct Color {
    float r;
    float g;
    float b;
    float a = {1.0f};
};

enum Location
{
    Host,
    Device
};

template<Location location>
struct Grid
{
    bool* alive;
    Color* color;

    Grid(int width, int height) {
        if (location == Location::Device) {
            CHECK_CUDA(cudaMalloc(&color, sizeof(Color) * width * height));
            CHECK_CUDA(cudaMalloc(&alive, sizeof(bool) * width * height));
        } else {
            color = new Color[width * height];
            alive = new bool[width * height];
        }
    }

    ~Grid() {
        if (location == Location::Device) {
            CHECK_CUDA(cudaFree(color));
            CHECK_CUDA(cudaFree(alive));
        } else {
            delete[] color;
            delete[] alive;
        }
    }
};


