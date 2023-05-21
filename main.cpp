// Include the necessary libraries
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <ctime>
#include <vector>

#define CELLS_NUM 10000
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define GRID_SIZE 100

struct Position {
    float x;
    float y;
};

// Not implemented yet
struct Velocity {
    float xVelocity;
    float yVelocity;
};

struct Cell {
    int color;
    struct Position position;
};

// Define the grid of cells
bool grid[GRID_SIZE][GRID_SIZE] = {false};

// Function to draw the grid
void DrawGrid() {
    // Clear the screen
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set the color to green
    glColor3f(0.0f, 1.0f, 0.0f);

    // Draw the cells
    glBegin(GL_QUADS);
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            if (grid[x][y]) {
                // Change coordinates from grid coordinates to OpenGL coordinates [-1, 1]
                float xPos = (float)x / GRID_SIZE * 2 - 1;
                float yPos = (float)y / GRID_SIZE * 2 - 1;
                float cellSize = 2.0f / GRID_SIZE;
                glVertex2f(xPos, yPos);
                glVertex2f(xPos + cellSize, yPos);
                glVertex2f(xPos + cellSize, yPos + cellSize);
                glVertex2f(xPos, yPos + cellSize);
            }
        }
    }
    glEnd();
}

bool isCellAlive(int x, int y) {
    return grid[x][y];
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
            if (isCellAlive(newX, newY)) {
                ++aliveNeighbors;
            }
        }
    }
    if (isCellAlive(x, y)) {
        return aliveNeighbors == 2 || aliveNeighbors == 3;
    } else {
        return aliveNeighbors == 3;
    }
}

// Function to update the positions of the cells
void UpdateGrid() {
    // Copy the current grid
    bool newGrid[GRID_SIZE][GRID_SIZE] = {false};
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            newGrid[x][y] = decideIfAlive(x, y);
        }
    }
    // Replace the old grid with the new grid
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            grid[x][y] = newGrid[x][y];
        }
    }
}

int main() {
    // Initialize the library
    if (!glfwInit()) {
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Cellular Automata", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize the grid with some random cells
    srand(time(NULL));
    for (int i = 0; i < CELLS_NUM; ++i) {
        int x = rand() % GRID_SIZE;
        int y = rand() % GRID_SIZE;
        grid[x][y] = true;
    }

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Get the current time
        static double lastTime = glfwGetTime();
        double currentTime = glfwGetTime();
        // If one second has passed since the last update
        if (currentTime - lastTime >= 0.9) {
            // Update the grid
            UpdateGrid();
            lastTime = currentTime;
        }

        // Draw the grid
        DrawGrid();

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
