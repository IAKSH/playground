#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <spdlog/spdlog.h>

#include "dataset.hpp"
#include "akaze.hpp"
#include "pca.hpp"

using namespace std;

void save_image(const vector<unsigned char>& images, int image_size, int index, const string& filename) {
    cv::Mat img(28, 28, CV_8UC1, (void*)&images[index * image_size]);
    cv::imwrite(filename, img);
}

int main() {
    spdlog::info("OpenCV version: {}", CV_VERSION);

    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");
    auto& train_data = fminist.get_train();


    std::vector<cv::Mat> descriptors_list;
    extract_akaze_features(train_data,descriptors_list,0,train_data.size());
    //extract_akaze_features_mt(train_data, descriptors_list, 4);

    // 进行 PCA 降维，假设降维到 50 个主成分
    int num_components = 50;
    cv::Mat reduced_descriptors = pca(descriptors_list, num_components);

    // 示例：展示 PCA 后的特征向量
    spdlog::info("Reduced descriptors shape: {} x {}", reduced_descriptors.rows, reduced_descriptors.cols);

    return 0;
}
