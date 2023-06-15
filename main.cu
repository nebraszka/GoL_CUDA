#include <Shader.hpp>
#include <GLFW/glfw3.h>

#include <display.hpp>
#include <data.hpp>
#include <cpu.hpp>
#include <gpu.cuh>

#include <getopt.h>
#include <chrono>

#define WIDTH 1920
#define HEIGHT 1080

#define SEED 271828

#define TIME_ITERATIONS 100

#define GPU_FAST_RENDER

#ifdef GPU_FAST_RENDER
#define setDataGPU setDataGPUFast
#else
#define setDataGPU setDataGPUSlow
#endif

void runCPU();
void runGPU();
void timeMeasureCPU(int iterations);
void timeMeasureGPU(int iterations);

int main(int argc, char** argv)
{
    int option = 0;

    while ((option = getopt(argc, argv, "cgtT")) != -1)
    {
        switch (option)
        {
            case 'c':
                runCPU();
                
            case 'g':
                runGPU();
                return 0;
            case 't':
                timeMeasureCPU(TIME_ITERATIONS);
                return 0;
            case 'T':
                timeMeasureGPU(TIME_ITERATIONS);
                return 0;
            default:
                std::cerr << "Nieznana opcja: " << option << "\n";
                return 1;
        }
    }

    std::cerr << "Proszę podać jedną z opcji: -c (CPU), -g (GPU), -t (pomiar czasu dla CPU), -T (pomiar czasu dla GPU))\n";

    return 1;
}

GLFWwindow* window;
void framebufferSizeCallback(GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }

void initializeGLFWAndGlad() {
    glfwInit();
    window = glfwCreateWindow(WIDTH, HEIGHT, "GOL & CUDA | Maja Nagarnowicz & Agnieszka Stefankowska", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
}

void initializeShaders() {
    Shader shader("../shader.vs", "../shader.fs");
    shader.use();
}

void runCPU() {

    initializeGLFWAndGlad();
    initializeShaders();

    Grid<Location::Host> gridA {WIDTH, HEIGHT};
    Grid<Location::Host> gridB {WIDTH, HEIGHT};

    gridA.randomInit(SEED);

    auto *current = &gridA;
    auto *next = &gridB;

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
    
        TempTexture texture {WIDTH, HEIGHT};
        texture.setDataCPU(current->color);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
    
        updateCPU(current, next, WIDTH, HEIGHT);
    
        std::swap(current->color, next->color);
        std::swap(current->alive, next->alive);
    
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}

void runGPU() {
    initializeGLFWAndGlad();
    initializeShaders();

    Grid<Location::Host> gridA {WIDTH, HEIGHT};
    Grid<Location::Host> gridB {WIDTH, HEIGHT};

    gridA.randomInit(SEED);

    auto *current = &gridA;
    auto *next = &gridB;

    Grid<Location::Device> currentGPU {WIDTH, HEIGHT};
    Grid<Location::Device> nextGPU {WIDTH, HEIGHT};

    CHECK_CUDA(cudaMemset(nextGPU.color, 0, sizeof(Color) * WIDTH * HEIGHT));
    CHECK_CUDA(cudaMemset(nextGPU.alive, 0, sizeof(bool) * WIDTH * HEIGHT));
    CHECK_CUDA(cudaMemcpy(currentGPU.color, current->color, sizeof(Color) * WIDTH * HEIGHT, cudaMemcpyDefault));
    CHECK_CUDA(cudaMemcpy(currentGPU.alive, current->alive, sizeof(bool) * WIDTH * HEIGHT, cudaMemcpyDefault));

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        TempTexture texture {WIDTH, HEIGHT};
        texture.setDataGPU(currentGPU.color);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);

        updateGPU(currentGPU.color, currentGPU.alive, nextGPU.color, nextGPU.alive, WIDTH, HEIGHT);
        std::swap(currentGPU.color, nextGPU.color);
        std::swap(currentGPU.alive, nextGPU.alive);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}

void timeMeasureGPU(int numIterations){
    initializeGLFWAndGlad();
    initializeShaders();

    Grid<Location::Host> gridA {WIDTH, HEIGHT};
    Grid<Location::Host> gridB {WIDTH, HEIGHT};

    gridA.randomInit(SEED);

    auto *current = &gridA;
    auto *next = &gridB;

    Grid<Location::Device> currentGPU {WIDTH, HEIGHT};
    Grid<Location::Device> nextGPU {WIDTH, HEIGHT};

    CHECK_CUDA(cudaMemset(nextGPU.color, 0, sizeof(Color) * WIDTH * HEIGHT));
    CHECK_CUDA(cudaMemset(nextGPU.alive, 0, sizeof(bool) * WIDTH * HEIGHT));
    CHECK_CUDA(cudaMemcpy(currentGPU.color, current->color, sizeof(Color) * WIDTH * HEIGHT, cudaMemcpyDefault));
    CHECK_CUDA(cudaMemcpy(currentGPU.alive, current->alive, sizeof(bool) * WIDTH * HEIGHT, cudaMemcpyDefault));


    float avg_time_gpu = 0.0;
    float totalTimeGpu = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    totalTimeGpu = 0;

    for (int i = 0; i < numIterations; ++i) {
        // Measure the calculation time
        cudaEventRecord(start);
        updateGPU(currentGPU.color, currentGPU.alive, nextGPU.color, nextGPU.alive, WIDTH, HEIGHT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTimeGpu += milliseconds;

        std::swap(currentGPU.color, nextGPU.color);
        std::swap(currentGPU.alive, nextGPU.alive);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    avg_time_gpu = totalTimeGpu / numIterations;
    std::cout << "GPU: średni czas obliczeń dla " << numIterations << " iteracji: " << avg_time_gpu << " ms" << std::endl;

    totalTimeGpu = 0;

    for(int i = 0; i < numIterations; ++i) {
        glClear(GL_COLOR_BUFFER_BIT);

        // Measure the rendering time
        cudaEventRecord(start);
        TempTexture texture {WIDTH, HEIGHT};
        texture.setDataGPU(currentGPU.color);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTimeGpu += milliseconds;
        updateGPU(currentGPU.color, currentGPU.alive, nextGPU.color, nextGPU.alive, WIDTH, HEIGHT);
        std::swap(currentGPU.color, nextGPU.color);
        std::swap(currentGPU.alive, nextGPU.alive);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    avg_time_gpu = totalTimeGpu / numIterations;
    std::cout << "GPU: średni czas renderowania dla " << numIterations << " iteracji: " << avg_time_gpu << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void timeMeasureCPU(int numIterations){
    initializeGLFWAndGlad();
    initializeShaders();

    Grid<Location::Host> gridA {WIDTH, HEIGHT};
    Grid<Location::Host> gridB {WIDTH, HEIGHT};

    gridA.randomInit(SEED);

    auto *current = &gridA;
    auto *next = &gridB;

    auto totalTime = 0;

    for (int i = 0; i < numIterations; ++i) {
        // Measure the calculation time
        auto start_time = std::chrono::high_resolution_clock::now();
        updateCPU(current, next, WIDTH, HEIGHT);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        totalTime += time/std::chrono::milliseconds(1);
    
        std::swap(current->color, next->color);
        std::swap(current->alive, next->alive);
    }

    auto avg_time = totalTime / numIterations;
    std::cout << "CPU: średni czas obliczeń dla " << numIterations << " iteracji: " << avg_time << " ms" << std::endl;

    for (int i = 0; i < numIterations; ++i) {
        glClear(GL_COLOR_BUFFER_BIT);

        // Measure the rendering time
        auto start_time = std::chrono::high_resolution_clock::now();

        TempTexture texture {WIDTH, HEIGHT};
        texture.setDataCPU(current->color);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        totalTime += time/std::chrono::milliseconds(1);
    
        updateCPU(current, next, WIDTH, HEIGHT);
    
        std::swap(current->color, next->color);
        std::swap(current->alive, next->alive);
    
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    avg_time = totalTime / numIterations;
    std::cout << "CPU: średni czas renderowania dla " << numIterations << " iteracji: " << avg_time << " ms" << std::endl;
}
