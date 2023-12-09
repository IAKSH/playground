#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <format>

#include <nioes/material.hpp>

nioes::Image::Image(std::string_view path,bool flip) noexcept(false) {
    load_from_file(path,flip);
}

nioes::Image::Image(unsigned char* source,int len,bool flip) noexcept {
    load_from_mem(source,len,flip);
}

nioes::Image::Image(unsigned char* source,int w,int h,int channel_num) noexcept 
    : width(w),height(h),channel_num(channel_num) {
    copy_from(source);
}

nioes::Image::Image(Image& img) noexcept 
    : width(img.get_width()), height(img.get_height()), channel_num(img.get_channel_num()) {
    copy_from(img.get_data());
}

nioes::Image::~Image() noexcept {
    free_data();
}

void nioes::Image::load_from_file(std::string_view path,bool flip) noexcept(false) {
    stbi_set_flip_vertically_on_load(flip);
    data = stbi_load(path.data(),&width,&height,&channel_num,0);
    if(!data)
        throw std::runtime_error(std::format("failed to load image from {}",path));
}

void nioes::Image::load_from_mem(unsigned char* source,int len,bool flip) noexcept {
    stbi_set_flip_vertically_on_load(flip);
    data = stbi_load_from_memory(source,len,&width,&height,&channel_num,0);
    if(!data)
        throw std::runtime_error("failed to load image from mem");
}

void nioes::Image::copy_from(unsigned char* source) noexcept {
    int size = width * height * channel_num;
    data = new unsigned char[size];
    std::copy(source,source + size,data);
}

void nioes::Image::free_data() noexcept {
    stbi_image_free(data);
}

unsigned char* nioes::Image::get_data() const noexcept {
    return data;
}

int nioes::Image::get_width() const noexcept {
    return width;
}

int nioes::Image::get_height() const noexcept {
    return height;
}

int nioes::Image::get_channel_num() const noexcept {
    return channel_num;
}

nioes::Texture::Texture(const Image& image,GLenum inner_format,bool rtti = true) noexcept(false)
    : width(image.get_width()),height(image.get_height()),inner_format(inner_format),rtti(rtti) {
    create_texture(image);
}

nioes::Texture::~Texture() noexcept {
    if(rtti) {
        free_vram();
    }
}

void nioes::Texture::create_texture(const Image& image) noexcept(false) {
    GLenum img_format;
    switch (image.get_channel_num())
    {
    case 1: img_format = GL_RED; break;
    case 3: img_format = GL_RGB; break;
    case 4: img_format = GL_RGBA; break;
    default:
        throw std::runtime_error(std::format("unkown image format with {} channels",image.get_channel_num()));
    } 

    glGenTextures(1,&id);
    glBindTexture(GL_TEXTURE_2D,id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,0,inner_format,width,height,0,img_format,GL_UNSIGNED_BYTE,image.get_data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,0);
}

void nioes::Texture::free_vram() noexcept {
    glDeleteTextures(1,&id);
}

GLuint nioes::Texture::get_id() const noexcept {
    return id;
}

int nioes::Texture::get_width() const noexcept {
    return width;
}

int nioes::Texture::get_height() const noexcept {
    return height;
}

GLenum nioes::Texture::get_inner_format() const noexcept {
    return inner_format;
}

nioes::CubeMap::CubeMap(int w,int h,GLenum inner_format) noexcept 
    : width(w),height(h),inner_format(inner_format) {
    alloc_vram();
}

nioes::CubeMap::~CubeMap() noexcept {
    free_vram();
}

void nioes::CubeMap::alloc_vram() noexcept {
    glGenTextures(1,&id);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP,id);
    // must pre-set the size of each texture2d in cubemap
    // or the fbo that this cubemap binding to will be incompete
    for (int i = 0; i < 6; i++) {
        if(inner_format == GL_DEPTH_COMPONENT) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, inner_format, width, height, 0, inner_format, GL_FLOAT, nullptr);
        }
        else {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, inner_format, width, height, 0, inner_format, GL_UNSIGNED_BYTE, nullptr);
        }
    }
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}

void nioes::CubeMap::free_vram() noexcept {
    glDeleteTextures(1, &id);
}

GLuint nioes::CubeMap::get_id() const noexcept {
    return id;
}

int nioes::CubeMap::get_width() const noexcept {
    return width;
}

int nioes::CubeMap::get_height() const noexcept {
    return height;
}

void nioes::CubeMap::load_from_image(Direction d,const Image& img) const noexcept {
    glBindTexture(GL_TEXTURE_CUBE_MAP,id);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    

    GLenum img_format;
    switch (img.get_channel_num())
    {
    case 1: img_format = GL_RED;  break;
    case 3: img_format = GL_RGB;  break;
    case 4: img_format = GL_RGBA; break;
    }

    glTexImage2D(static_cast<GLenum>(d),0,inner_format,width,height,0,img_format,GL_UNSIGNED_BYTE,img.get_data());
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP,0);
}

void nioes::CubeMap::load_from_file(Direction d,std::string_view path) const noexcept {
    load_from_image(d,Image(path));
}
