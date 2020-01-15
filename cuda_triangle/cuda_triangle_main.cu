#include "cuda_triangle_kernel.h"
#include "image_writer.hpp"

#include <iostream>
#include <chrono>

int main()
{
    // Triangle vertic
    float vertices[] = {-0.5f, -0.5f, 0.0f
                        ,0.5f, -0.5f, 0.0f
                        ,0.0f,  0.5f, 0.0f};

    // host memory
    // pixel postion
    vec2_t p0 = {(vertices[0] + 1)*WIDTH/2, (vertices[1] + 1)*HEIGHT/2}; 
    vec2_t p1 = {(vertices[3] + 1)*WIDTH/2, (vertices[4] + 1)*HEIGHT/2}; 
    vec2_t p2 = {(vertices[6] + 1)*WIDTH/2, (vertices[7] + 1)*HEIGHT/2}; 
    // color
    pixel_t c0 = {(unsigned char)(1.0 * 255), (unsigned char)(0.0 * 255), (unsigned char)(0.0 * 255)};
    pixel_t c1 = {(unsigned char)(0.0 * 255), (unsigned char)(1.0 * 255), (unsigned char)(0.0 * 255)};
    pixel_t c2 = {(unsigned char)(0.0 * 255), (unsigned char)(0.0 * 255), (unsigned char)(1.0 * 255)};
    
    // time stamp start
    auto t1 = std::chrono::high_resolution_clock::now();

    // device memory
    vec2_t *d_p0, *d_p1, *d_p2;
    pixel_t *d_c0, *d_c1, *d_c2;
    cudaMalloc((void **)&d_p0, sizeof(vec2_t));
    cudaMalloc((void **)&d_p1, sizeof(vec2_t));
    cudaMalloc((void **)&d_p2, sizeof(vec2_t));
    cudaMalloc((void **)&d_c0, sizeof(pixel_t));
    cudaMalloc((void **)&d_c1, sizeof(pixel_t));
    cudaMalloc((void **)&d_c2, sizeof(pixel_t));
    cudaMemcpy(d_p0, &p0, sizeof(vec2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1, &p1, sizeof(vec2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2, &p2, sizeof(vec2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c0, &c0, sizeof(pixel_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1, &c1, sizeof(pixel_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2, &c2, sizeof(pixel_t), cudaMemcpyHostToDevice);
    // output image
    pixel_t img[WIDTH*HEIGHT] = {0, };
    pixel_t *d_img;
    cudaMalloc((void **)&d_img, sizeof(pixel_t)*WIDTH*HEIGHT);

    draw_triangle<<<HEIGHT, WIDTH>>>(d_c0, d_c1, d_c2, d_p0, d_p1, d_p2, d_img);

    cudaDeviceSynchronize();
    cudaMemcpy(img, d_img, WIDTH*HEIGHT*sizeof(pixel_t), cudaMemcpyDeviceToHost);
    
    // time stamp end
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exe_time = t2 - t1;
    std::cout << "execution time : " << exe_time.count() << " ms" << std::endl;

    write_image("test.png", WIDTH, HEIGHT, img);

    return 0;
}