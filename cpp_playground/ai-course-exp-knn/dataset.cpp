#include "dataset.hpp"
#include <fstream>
#include <format>
#include <spdlog/spdlog.h>
#include <stdexcept>

std::vector<unsigned char> FashionMINIST::readMNISTImages(std::string_view filepath, int &number_of_images, int &image_size) {
    std::ifstream file(filepath.data(), std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + std::string(filepath));
    }

    int magic_number = 0, number_of_rows = 0, number_of_columns = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2051) {  // Expected magic number for image files
        throw std::runtime_error("Invalid magic number in image file: " + std::string(filepath));
    }

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = __builtin_bswap32(number_of_images);

    file.read((char*)&number_of_rows, sizeof(number_of_rows));
    number_of_rows = __builtin_bswap32(number_of_rows);

    file.read((char*)&number_of_columns, sizeof(number_of_columns));
    number_of_columns = __builtin_bswap32(number_of_columns);

    image_size = number_of_rows * number_of_columns;
    std::vector<unsigned char> images(number_of_images * image_size);
    file.read((char*)images.data(), images.size());

    return images;
}

std::vector<unsigned char> FashionMINIST::readMNISTLabels(std::string_view filepath, int &number_of_labels) {
    std::ifstream file(filepath.data(), std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + std::string(filepath));
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2049) {  // Expected magic number for label files
        throw std::runtime_error("Invalid magic number in label file: " + std::string(filepath));
    }

    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = __builtin_bswap32(number_of_labels);

    std::vector<unsigned char> labels(number_of_labels);
    file.read((char*)labels.data(), labels.size());

    return labels;
}

void FashionMINIST::load_dataset(DataVec& vec, std::string_view path, std::string_view image_filename, std::string_view label_filename) {
    int number_of_images, image_size;
    std::vector<unsigned char> images = readMNISTImages(std::format("{}/raw/{}", path, image_filename), number_of_images, image_size);

    int number_of_labels;
    std::vector<unsigned char> labels = readMNISTLabels(std::format("{}/raw/{}", path, label_filename), number_of_labels);

    if (number_of_images != number_of_labels) {
        throw std::runtime_error("Number of images does not match number of labels");
    }

    for (int i = 0; i < number_of_images; i++) {
        // 使用深拷贝
        cv::Mat img(IMAGE_H, IMAGE_W, CV_8UC1);
        std::memcpy(img.data, &images[i * image_size], image_size);
        int label = labels[i];
        vec.emplace_back(img, label);
    }
}

FashionMINIST::FashionMINIST(std::string_view path) {
    load_dataset(train, path, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    load_dataset(val, path, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
}


const FashionMINIST::DataVec& FashionMINIST::get_train() const {
    return train;
}

const FashionMINIST::DataVec& FashionMINIST::get_val() const {
    return val;
}
