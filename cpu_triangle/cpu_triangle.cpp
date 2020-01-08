#include <iostream>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// resolution
#define WIDTH 1000
#define HEIGHT 500

typedef struct pixel_t
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} pixel_t;

typedef struct vec2_t
{
    float x;
    float y;
} vec2_t;

bool point_inside_trigon(vec2_t p, vec2_t p0, vec2_t p1, vec2_t p2)
{
    float area = 0.5 *(-p1.y*p2.x + p0.y*(-p1.x + p2.x) + p0.x*(p1.y - p2.y) + p1.x*p2.y);

    float s = 1/(2*area)*(p0.y*p2.x - p0.x*p2.y + (p2.y - p0.y)*p.x + (p0.x - p2.x)*p.y);
    float t = 1/(2*area)*(p0.x*p1.y - p0.y*p1.x + (p0.y - p1.y)*p.x + (p1.x - p0.x)*p.y);

    if(s>0 && t>0 && (1-s-t)>0)
        return true;
    else
        return false;
}

pixel_t color_interpolation(pixel_t c0, pixel_t c1, pixel_t c2, vec2_t p, vec2_t p0, vec2_t p1, vec2_t p2)
{
    pixel_t out;

    float d0 = sqrt((p0.x - p.x)*(p0.x - p.x) + (p0.y - p.y)*(p0.y - p.y));
    float d1 = sqrt((p1.x - p.x)*(p1.x - p.x) + (p1.y - p.y)*(p1.y - p.y));
    float d2 = sqrt((p2.x - p.x)*(p2.x - p.x) + (p2.y - p.y)*(p2.y - p.y));

    float d0_inv = 1/d0;
    float d1_inv = 1/d1;
    float d2_inv = 1/d2;

    float w0 = d0_inv/(d0_inv+d1_inv+d2_inv);
    float w1 = d1_inv/(d0_inv+d1_inv+d2_inv);
    float w2 = d2_inv/(d0_inv+d1_inv+d2_inv);

    out.r = w0*c0.r + w1*c1.r + w2*c2.r;
    out.g = w0*c0.g + w1*c1.g + w2*c2.g;
    out.b = w0*c0.b + w1*c1.b + w2*c2.b;

    return out;
}

int main()
{
    // Triangle vertic
    float vertices[] = {-0.5f, -0.5f, 0.0f
                        ,0.5f, -0.5f, 0.0f
                        ,0.0f,  0.5f, 0.0f};

    pixel_t img[WIDTH*HEIGHT] = {0, };

    // pixel postion
    vec2_t p0 = {(vertices[0] + 1)*WIDTH/2, (vertices[1] + 1)*HEIGHT/2}; 
    vec2_t p1 = {(vertices[3] + 1)*WIDTH/2, (vertices[4] + 1)*HEIGHT/2}; 
    vec2_t p2 = {(vertices[6] + 1)*WIDTH/2, (vertices[7] + 1)*HEIGHT/2}; 
    // color
    pixel_t c0 = {(unsigned char)(1.0 * 255), (unsigned char)(0.0 * 255), (unsigned char)(0.0 * 255)};
    pixel_t c1 = {(unsigned char)(0.0 * 255), (unsigned char)(1.0 * 255), (unsigned char)(0.0 * 255)};
    pixel_t c2 = {(unsigned char)(0.0 * 255), (unsigned char)(0.0 * 255), (unsigned char)(1.0 * 255)};

    // draw
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < HEIGHT; i++)
    {
        for(int j = 0; j < WIDTH; j++)
        {
            vec2_t p = {(float)j, (float)i};
            bool is_inside = point_inside_trigon(p, p0, p1, p2);
            if(is_inside)
            {
                pixel_t intp = color_interpolation(c0, c1, c2, p, p0, p1, p2);
                img[i * WIDTH + j].r = intp.r;
                img[i * WIDTH + j].g = intp.g;
                img[i * WIDTH + j].b = intp.b;
            }
            else
            {
                img[i * WIDTH + j].r = 255 * 0.2;
                img[i * WIDTH + j].g = 255 * 0.3;
                img[i * WIDTH + j].b = 255 * 0.3;
            }
        }
    }

    stbi_flip_vertically_on_write(1);
    stbi_write_png("test.png", WIDTH, HEIGHT, 3, img, 0);

    return 0;
}