#include <Shader.hpp>
#include <GLFW/glfw3.h>

#include <display.hpp>
#include <data.hpp>
#include <cpu.hpp>
#include <gpu.cuh>

constexpr int WIDTH = 1920;
constexpr int HEIGHT = 1080;

void framebufferSizeCallback(GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }

int main()
{
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "My Window", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    Shader shader("../shader.vs", "../shader.fs");
    shader.use();

    Grid<Location::Host> gridA {WIDTH, HEIGHT};
    Grid<Location::Host> gridB {WIDTH, HEIGHT};

    auto* current = &gridA;
    auto* next = &gridB;


    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            current->alive[y * WIDTH + x] = rand() % 2;
            current->color[y * WIDTH + x] = {
                (rand() % 255) / 255.0f,
                (rand() % 255) / 255.0f,
                (rand() % 255) / 255.0f
            };
        }
    }

    Grid<Location::Device> currentGPU {WIDTH, HEIGHT};
    Grid<Location::Device> nextGPU {WIDTH, HEIGHT};
    CHECK_CUDA(cudaMemset(nextGPU.color, 0, sizeof(Color) * WIDTH * HEIGHT));
    CHECK_CUDA(cudaMemset(nextGPU.alive, 0, sizeof(bool) * WIDTH * HEIGHT));
    CHECK_CUDA(cudaMemcpy(currentGPU.color, current->color, sizeof(Color) * WIDTH * HEIGHT, cudaMemcpyDefault));
    CHECK_CUDA(cudaMemcpy(currentGPU.alive, current->alive, sizeof(bool) * WIDTH * HEIGHT, cudaMemcpyDefault));


    // while (!glfwWindowShouldClose(window)) {
    //     glClear(GL_COLOR_BUFFER_BIT);
    
    //     TempTexture texture {WIDTH, HEIGHT};
    //     texture.setDataCPU(current->color);
    //     glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
    
    //     updateCPU(current, next, WIDTH, HEIGHT);
    
    //     std::swap(current->color, next->color);
    //     std::swap(current->alive, next->alive);
    
    //     glfwSwapBuffers(window);
    //     glfwPollEvents();
    // }

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        TempTexture texture {WIDTH, HEIGHT};
        texture.setDataGPUFast(currentGPU.color);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);

        updateGPU(currentGPU.color, currentGPU.alive, nextGPU.color, nextGPU.alive, WIDTH, HEIGHT);
        std::swap(currentGPU.color, nextGPU.color);
        std::swap(currentGPU.alive, nextGPU.alive);

        // Color* tmpColor = nextGPU.color;
        // bool* tmpAlive = nextGPU.alive;
        // nextGPU.color = currentGPU.color;
        // nextGPU.alive = currentGPU.alive;
        // currentGPU.color = tmpColor;
        // currentGPU.alive = tmpAlive;

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}