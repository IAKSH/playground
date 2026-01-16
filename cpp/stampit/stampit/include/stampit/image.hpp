#pragma once
#include <cstdint>
#include <string_view>
#include <opencv4/opencv2/opencv.hpp>

namespace stampit {
    struct Rect {
        int x,y,w,h;
    };

    class Image {
    public:
        Image(std::string_view path);
        Image(const Image& img);
        Image(cv::Mat mat);
        Image() = default;
        ~Image() = default;
        Image copy();
        void load(std::string_view path);
        void save(std::string_view path) const;
        void save_with_transparent_pix(std::string_view path,int x,int y,int w,int h) const;
        bool empty() const;
        int width() const;
        int height() const;
        size_t step() const;
        uchar* data() const;
        void threshold(uint8_t th);
        void denoise(int min_area);
        void set_color(uint8_t r,uint8_t g,uint8_t b);
        Rect get_min_boundary(int margin = 0) const;
        cv::Mat& __mat{mat};
        Image& operator=(Image& img);
    private:
        cv::Mat mat;
    };
}