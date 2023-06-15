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
    int width, height;

    Grid(int width, int height) {
        this->width = width;
        this->height = height;

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

    void randomInit(int seed){
        srand(seed);
        for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            alive[y * width + x] = rand() % 2;
            color[y * width + x] = {
                (rand() % 255) / 255.0f,
                (rand() % 255) / 255.0f,
                (rand() % 255) / 255.0f
            };
        }
    }
    }
};

template <Location location>
struct GridPointers {
    Grid<location>* current;
    Grid<location>* next;
};
