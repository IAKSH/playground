// Image,Texture,Phong-bling Lighting arg

#include <glad/gles2.h>
#include <stb_image.h>
#include <vector>

namespace nioes {
    // 储存在RAM的图片，全局唯一
    class Image {

    };

    // 储存在VRAM的图片，全局唯一
    // 如果需要实现某一Object实例动态更改其texture内容，请考虑使用图层覆盖
    // 而不是直接修改原图层
    class Texture {

    };

    // Blinn-Phone光照模型
    // 物体的受光照参数
    class LightingPara {

    };

    // 阴影贴图参数
    // 物体的受阴影参数
    class ShadowPara {

    };

    class Material {
    public:
        std::vector<Texture> textures;
        LightingPara lighting;
        ShadowPara shadow;
    };
}