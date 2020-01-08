#include "image_writer.hpp"
#include "common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void write_image(const char* name, int width, int height, void* img)
{
    stbi_flip_vertically_on_write(1);
    stbi_write_png(name, width, height, 3, img, 0);
}