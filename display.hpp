#pragma once

#include <Shader.hpp>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>


#include <data.hpp>

// Temporary texture that gets bound in ctor and unbound in dtor
struct TempTexture
{
    TempTexture(int width, int height) : width(width), height(height), resource(nullptr)
    {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   // Set texture wrapping to GL_REPEAT
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Pre-allocate texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

        // glGenerateMipmap(GL_TEXTURE_RECTANGLE);

    }

    ~TempTexture() {
        if (resource != nullptr) {
            cudaGraphicsUnregisterResource(resource);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        glDeleteTextures(1, &textureID);
    }

    void setDataCPU(Color* img)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, img);
    }

    void setDataGPUSlow(Color* img)
    {
        std::vector<Color> hostData;
        hostData.resize(width * height);
        CHECK_CUDA(cudaMemcpy(hostData.data(), img, sizeof(Color) * width * height, cudaMemcpyDefault));
        setDataCPU(hostData.data());
    }

void setDataGPUFast(Color* img) {

        CHECK_CUDA(cudaGraphicsGLRegisterImage(&resource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    cudaArray_t array;
    CHECK_CUDA(cudaGraphicsMapResources(1, &resource));
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    CHECK_CUDA(cudaMemcpy2DToArray(array, 0, 0, img, width * sizeof(Color), width * sizeof(Color), height, cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaGraphicsUnmapResources(1, &resource));
}

 private:
    int width;
    int height;
    GLuint textureID;
    cudaGraphicsResource_t resource;
};
