#pragma once

#include <data.hpp>

void updateCPU(Grid<Location::Host>* current, Grid<Location::Host>* next, int width, int height)
{
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {

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
                    if (current->alive[newY * width + newX]) {
                        r += current->color[newY * width + newX].r;
                        g += current->color[newY * width + newX].g;
                        b += current->color[newY * width + newX].b;
                        ++aliveNeighbours;
                    }
                }
            }

            bool alive = current->alive[y * width +x]
                       ? (aliveNeighbours == 2 || aliveNeighbours == 3)
                       : (aliveNeighbours == 3);
            next->alive[y * width + x] = alive;

            if (alive) {
                // If the cell is alive in the next generation, inherit the color from its neighbors
                next->color[y * width + x] = {r / aliveNeighbours, g / aliveNeighbours, b / aliveNeighbours};
            }
            else {
                next->color[y * width + x] = {0, 0, 0};
            }
        }
    }
}
