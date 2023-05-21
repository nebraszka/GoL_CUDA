#include <GLFW/glfw3.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstring>

#define CELLS_NUM 10000
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define GRID_SIZE 500

struct Position {
    float x;
    float y;
};

struct Color {
    float r;
    float g;
    float b;
};

struct Cell {
    Color color;
    Position position;
    bool alive;
};

Cell grid[GRID_SIZE][GRID_SIZE] = {{{0}}};

void DrawGrid() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            if (grid[x][y].alive) {
                float xPos = (float)x / GRID_SIZE * 2 - 1;
                float yPos = (float)y / GRID_SIZE * 2 - 1;
                float cellSize = 2.0f / GRID_SIZE;
                glColor3f(grid[x][y].color.r, grid[x][y].color.g, grid[x][y].color.b);
                glVertex2f(xPos, yPos);
                glVertex2f(xPos + cellSize, yPos);
                glVertex2f(xPos + cellSize, yPos + cellSize);
                glVertex2f(xPos, yPos + cellSize);
            }
        }
    }
    glEnd();
}

bool decideIfAlive(int x, int y) {
    int aliveNeighbors = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            int newX = (x + dx + GRID_SIZE) % GRID_SIZE;
            int newY = (y + dy + GRID_SIZE) % GRID_SIZE;
            if (grid[newX][newY].alive) {
                ++aliveNeighbors;
            }
        }
    }
    return grid[x][y].alive ? (aliveNeighbors == 2 || aliveNeighbors == 3) : (aliveNeighbors == 3);
}

Color inheritColor(int x, int y) {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    int aliveNeighbours = 0;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            int newX = (x + dx + GRID_SIZE) % GRID_SIZE;
            int newY = (y + dy + GRID_SIZE) % GRID_SIZE;
            if (grid[newX][newY].alive) {
                r += grid[newX][newY].color.r;
                g += grid[newX][newY].color.g;
                b += grid[newX][newY].color.b;
                aliveNeighbours++;
            }
        }
    }
    if (aliveNeighbours == 0) {
        return {0.0f, 0.0f, 0.0f};
    }

    return {r / aliveNeighbours, g / aliveNeighbours, b / aliveNeighbours};
}

void UpdateGrid() {
    Cell newGrid[GRID_SIZE][GRID_SIZE] = {{{0}}};
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            newGrid[x][y].alive = decideIfAlive(x, y);
            if(newGrid[x][y].alive) {
                // If the cell is alive in the next generation, inherit the color from its neighbors
                newGrid[x][y].color = inheritColor(x, y);
            }
        }
    }
    memcpy(grid, newGrid, sizeof(newGrid));
}


int main() {
    if (!glfwInit()) {
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "GoL Maja Nagarnowicz Agnieszka Stefankowska", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    srand(time(NULL));
    for (int i = 0; i < CELLS_NUM; ++i) {
        int x = rand() % GRID_SIZE;
        int y = rand() % GRID_SIZE;
        grid[x][y].alive = true;
        // Set the color to some initial value
        grid[x][y].color.r = (rand() % 255) / 255.0f;
        grid[x][y].color.g = (rand() % 255) / 255.0f;
        grid[x][y].color.b = (rand() % 255) / 255.0f;
    }
    while (!glfwWindowShouldClose(window)) {
        static double lastTime = glfwGetTime();
        double currentTime = glfwGetTime();
        if (currentTime - lastTime >= 0.05) {
            UpdateGrid();
            lastTime = currentTime;
        }

        DrawGrid();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
