#pragma once

#include <cuda_runtime.h>
#include <data.hpp>

__global__ void updateGPU_kernel(Color* currentColor, bool* currentAlive, Color* nextColor, bool* nextAlive, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        int aliveNeighbours = 0;

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                int newX = (x + dx + width) % width;
                int newY = (y + dy + height) % height;
                if (currentAlive[newY * width + newX]) {
                    r += currentColor[newY * width + newX].r;
                    g += currentColor[newY * width + newX].g;
                    b += currentColor[newY * width + newX].b;
                    ++aliveNeighbours;
                }
            }
        }

        bool alive = currentAlive[y * width + x]
                     ? (aliveNeighbours == 2 || aliveNeighbours == 3)
                     : (aliveNeighbours == 3);
        nextAlive[y * width + x] = alive;

        if (alive) {
            // If the cell is alive in the next generation, inherit the color from its neighbors
            nextColor[y * width + x] = {r / aliveNeighbours, g / aliveNeighbours, b / aliveNeighbours};
        }
        else {
            nextColor[y * width + x] = {0, 0, 0};
        }
    }
}

void updateGPU(Color* currentColor, bool* currentAlive, Color* nextColor, bool* nextAlive, int width, int height) {
    dim3 blockSize(16, 16);  // Set the block size
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);  // Set the grid size
    updateGPU_kernel<<<gridSize, blockSize>>>(currentColor, currentAlive, nextColor, nextAlive, width, height);
    CHECK_CUDA(cudaDeviceSynchronize()); // wait for the kernel to finish
}

