#include "cuda_triangle_kernel.h"

#include <cmath>

__device__ void point_inside_trigon(vec2_t p, vec2_t p0, vec2_t p1, vec2_t p2, bool* is_inside)
{
    float area = 0.5 *(-p1.y*p2.x + p0.y*(-p1.x + p2.x) + p0.x*(p1.y - p2.y) + p1.x*p2.y);

    float s = 1/(2*area)*(p0.y*p2.x - p0.x*p2.y + (p2.y - p0.y)*p.x + (p0.x - p2.x)*p.y);
    float t = 1/(2*area)*(p0.x*p1.y - p0.y*p1.x + (p0.y - p1.y)*p.x + (p1.x - p0.x)*p.y);

    if(s>0 && t>0 && (1-s-t)>0)
        *is_inside = true;
    else
        *is_inside = false;
}

__device__ void color_interpolation(pixel_t c0, pixel_t c1, pixel_t c2, vec2_t p, vec2_t p0, vec2_t p1, vec2_t p2, pixel_t* out)
{
    float d0 = sqrt((p0.x - p.x)*(p0.x - p.x) + (p0.y - p.y)*(p0.y - p.y));
    float d1 = sqrt((p1.x - p.x)*(p1.x - p.x) + (p1.y - p.y)*(p1.y - p.y));
    float d2 = sqrt((p2.x - p.x)*(p2.x - p.x) + (p2.y - p.y)*(p2.y - p.y));

    float d0_inv = 1/d0;
    float d1_inv = 1/d1;
    float d2_inv = 1/d2;

    float w0 = d0_inv/(d0_inv+d1_inv+d2_inv);
    float w1 = d1_inv/(d0_inv+d1_inv+d2_inv);
    float w2 = d2_inv/(d0_inv+d1_inv+d2_inv);

    out->r = w0*c0.r + w1*c1.r + w2*c2.r;
    out->g = w0*c0.g + w1*c1.g + w2*c2.g;
    out->b = w0*c0.b + w1*c1.b + w2*c2.b;
}

__global__ void draw_triangle(pixel_t *c0, pixel_t *c1, pixel_t *c2, vec2_t *p0, vec2_t *p1, vec2_t *p2, pixel_t* img)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    vec2_t p = {(float)j, (float)i};
    bool is_inside;
    point_inside_trigon(p, *p0, *p1, *p2, &is_inside);
    if(is_inside)
    {
        pixel_t intp;
        color_interpolation(*c0, *c1, *c2, p, *p0, *p1, *p2, &intp);
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