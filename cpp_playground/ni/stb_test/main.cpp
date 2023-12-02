extern "C"
{
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
}

#include <iostream>
#include <string>
#include <vector>
#include <cinttypes>

class Image
{
private:
    uint8_t *data = NULL;
    int width, hight, channel_count;

public:
    Image(){};
    Image(const char *path)
    {
        open(path);
    }

    ~Image()
    {
        stbi_image_free(data);
    }

    void open(const char *path)
    {
        if (data)
            stbi_image_free(data);
        data = stbi_load(path, &width, &hight, &channel_count, 0);
    }

    void resize(int new_width, int new_hight)
    {
        uint8_t *new_data = new uint8_t[new_width * new_hight * channel_count];
        stbir_resize(data, width, hight, 0, new_data, new_width, new_hight, 0, STBIR_TYPE_UINT8, channel_count, STBIR_ALPHA_CHANNEL_NONE, 0,
                     STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                     STBIR_COLORSPACE_SRGB, nullptr);
        stbi_image_free(data);
        data = new_data;
        width = new_width;
        hight = new_hight;
    }

    void save(const char *path)
    {
        stbi_write_png(path, width, hight, channel_count, data, 0);
    }

    uint8_t *getData()
    {
        return data;
    }
};

int main()
{
    std::cout << "Hello, stb_Image" << std::endl;
    Image image_in("../a.png");
    image_in.resize(512, 512);
    image_in.save("../out.png");
    stbi_write_png("../b.png",50,50,4,image_in.getData(),0);
    return 0;
}