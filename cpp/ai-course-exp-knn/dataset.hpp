#pragma once

#include <vector>
#include <string_view>
#include <opencv2/opencv.hpp>

class FashionMINIST {
    using DataVec = std::vector<std::pair<cv::Mat,int>>;
public:
    FashionMINIST(std::string_view path);
    FashionMINIST(FashionMINIST&) = delete;
    ~FashionMINIST() = default;

    const DataVec& get_train() const;
    const DataVec& get_val() const;

    static constexpr int IMAGE_W = 28;
    static constexpr int IMAGE_H = 28;

private:
    DataVec train,val;
    std::vector<unsigned char> readMNISTImages(std::string_view filepath, int &number_of_images, int &image_size);
    std::vector<unsigned char> readMNISTLabels(std::string_view filepath, int &number_of_labels);
    void load_dataset(DataVec& vec, std::string_view path, std::string_view image_filename, std::string_view label_filename);
};