#pragma once

#include "common.h"

__global__ void draw_triangle(pixel_t *c0, pixel_t *c1, pixel_t *c2, vec2_t *p0, vec2_t *p1, vec2_t *p2, pixel_t* img);
