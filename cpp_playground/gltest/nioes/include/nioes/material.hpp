// Image,Texture,Phong-bling Lighting arg

#include <glad/gles2.h>
#include <glm/vec4.hpp>
#include <stb_image.h>
#include <string_view>
#include <vector>
#include <memory>
#include <format>

namespace nioes {
    // 储存在RAM的图片，全局唯一
    class Image {
    private:
        unsigned char* data;
        int width;
        int height;
        int channel_num;
        void load_from_file(std::string_view path,bool flip) noexcept(false);
        void load_from_mem(unsigned char* source,int len,bool flip) noexcept(false);
        void copy_from(unsigned char* source) noexcept;
        void free_data() noexcept;

    public:
        Image(std::string_view path,bool flip = true) noexcept(false);
        Image(unsigned char* source,int len,bool flip = true) noexcept;
        Image(unsigned char* source,int w,int h,int channel_num) noexcept;
        Image(Image&) noexcept;
        ~Image() noexcept;

        unsigned char* get_data() const noexcept;
        int get_width() const noexcept;
        int get_height() const noexcept;
        int get_channel_num() const noexcept;
    };

    // 储存在VRAM的图片，全局唯一
    // 如果需要实现某一Object实例动态更改其texture内容，请考虑使用图层覆盖
    // 而不是直接修改原图层
    class Texture {
    protected:
        GLuint id;
        int width;
        int height;
        const GLenum inner_format;
        const bool rtti;
        // 如果以后需要改纹理的过滤方式，或者其他类似的东西
        // 可以考虑把这个函数改成virtual，然后在子类中override
        // 最后由该子类创建各inner format的Texture
        void create_texture(const Image& image) noexcept(false);
        void free_vram() noexcept;
    
    public:
        Texture(const Image& image,GLenum inner_format,bool rtti = true) noexcept(false);
        Texture(Texture&) = delete;
        ~Texture() noexcept;

        GLuint get_id() const noexcept;
        int get_width() const noexcept;
        int get_height() const noexcept;
        GLenum get_inner_format() const noexcept;
    };

    // 实际上只有Red通道
    class GrayscaleTexture : public Texture {
    public:
        GrayscaleTexture(const Image& image,bool rtti = true) noexcept
            : Texture(image,GL_R8,rtti) {}
    };

    // 实际上只有Red和Green通道
    class GrayScaleAlphaTexture : public Texture {
    public:
        GrayScaleAlphaTexture(const Image& image,bool rtti = true) noexcept
            : Texture(image,GL_RG8,rtti) {}
    };

    class RBGTexture : public Texture {
    public:
        RBGTexture(const Image& image,bool rtti = true) noexcept 
            : Texture(image,GL_RGB8,rtti) {}
    };

    class RGBATexture : public Texture {
    public:
        RGBATexture(const Image& image,bool rtti = true) noexcept 
            : Texture(image,GL_RGBA,rtti) {}
    };

    class SRGBTexture : public Texture {
    public:
        SRGBTexture(const Image& image,bool rtti = true) noexcept 
            : Texture(image,GL_SRGB8,rtti) {}
    };

    class SRGBATexture : public Texture {
    public:
        SRGBATexture(const Image& image,bool rtti = true) noexcept 
            : Texture(image,GL_SRGB8_ALPHA8,rtti) {}
    };

    class HDRTexture : public Texture {
    public:
        HDRTexture(const Image& image,bool rtti = true) noexcept 
            : Texture(image,GL_RGBA16F,rtti) {}
    };

    class DepthTexture : public Texture {
    public:
        DepthTexture(const Image& image,bool rtti = true) noexcept 
            : Texture(image,GL_DEPTH_COMPONENT24,rtti) {}
    };

    class StencilTexture : public Texture {
    public:
        StencilTexture(const Image& image,bool rtti = true) noexcept 
            : Texture(image,GL_STENCIL_INDEX,rtti) {}
    };

    template <typename T>
    requires std::derived_from<T,Texture>
    std::unique_ptr<T> load_texture_from_file(std::string_view path) noexcept(false)
    {
        // TODO: 创建Image后完美转发到T构造
        // 改了texture的继承结构以后，好像意义不大了，但是还是留着吧
        return std::make_unique<T>(Image(path));
    }

    class CubeMap {
    private:
        GLuint id;
        int width;
        int height;
        const GLenum inner_format;
        void alloc_vram() noexcept;
        void free_vram() noexcept;
        
    public:
        CubeMap(int w,int h,GLenum inner_format) noexcept;
        CubeMap(CubeMap&) = delete;
        ~CubeMap() noexcept;

        enum class Direction {
            NegativeX = GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
            NegativeY = GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
            NegativeZ = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
            PositiveX = GL_TEXTURE_CUBE_MAP_POSITIVE_X,
            PositiveY = GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
            PositiveZ = GL_TEXTURE_CUBE_MAP_POSITIVE_Z
        };

        GLuint get_id() const noexcept;
        int get_width() const noexcept;
        int get_height() const noexcept;
        void load_from_image(Direction d,const Image& img) const noexcept;
        void load_from_file(Direction d,std::string_view path) const noexcept;
    };

    class SRGBCubeMap : public CubeMap {
    public:
        SRGBCubeMap(int w,int h) noexcept
            : CubeMap(w,h,GL_SRGB8) {}
    };

    class SRGBACubeMap : public CubeMap {
    public:
        SRGBACubeMap(int w,int h) noexcept
            : CubeMap(w,h,GL_SRGB8_ALPHA8) {}
    };

    class HDRCubeMap : public CubeMap {
    public:
        HDRCubeMap(int w,int h) noexcept
            : CubeMap(w,h,GL_RGBA16F) {}
    };

    class DepthCubeMap : public CubeMap {
    public:
        DepthCubeMap(int w,int h) noexcept
            : CubeMap(w,h,GL_DEPTH_COMPONENT24) {}
    };

    // Blinn-Phone光照模型
    // 物体的受光照参数
    class LightingPara {
    public:
        glm::vec4 diffuse;
        glm::vec4 specular;
        glm::vec4 ambient;
    };

    // 阴影贴图参数
    // 物体的受阴影参数
    /*
    class ShadowPara {
        
    };
    */

    // 材质，作用对象至少是单个Mesh，至多是一整个Model
    class Material {
    public:
        std::vector<Texture> textures;
        LightingPara lighting;
        //ShadowPara shadow;
    };
}