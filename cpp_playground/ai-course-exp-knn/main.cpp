#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <spdlog/spdlog.h>

#include "dataset.hpp"

using namespace std;

void save_image(const vector<unsigned char>& images, int image_size, int index, const string& filename) {
    cv::Mat img(28, 28, CV_8UC1, (void*)&images[index * image_size]);
    cv::imwrite(filename, img);
}

int main() {
    spdlog::info("OpenCV version: {}",CV_VERSION);

    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");
    auto& v = fminist.get_train();
    auto& aa = v[0].first;
    for(int i = 0;i < 5;i++) {
        spdlog::warn(v[i].first.empty() ? "null" : "not null");
        cv::imshow(std::format("img_{}",i),v[i].first);
        spdlog::warn("label: {}",v[i].second);
    }

    cv::waitKey(0);
    return 0;
}
