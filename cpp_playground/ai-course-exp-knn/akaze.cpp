#include "akaze.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <mutex>

using namespace cv;

// 用于线程安全的特征描述符容器
static std::mutex descriptor_mutex;
static Ptr<AKAZE> akaze = AKAZE::create();

void extract_akaze_features(const std::vector<std::pair<Mat, int>>& data, std::vector<Mat>& result, int start, int end) {
    for (int i = start; i < end; ++i) {
        Mat descriptors;
        std::vector<KeyPoint> keypoints;

        akaze->detectAndCompute(data[i].first, noArray(), keypoints, descriptors);

        std::lock_guard<std::mutex> lock(descriptor_mutex);
        result.push_back(descriptors);
    }
}

void extract_akaze_features_mt(const std::vector<std::pair<Mat, int>>& data, std::vector<Mat>& result, int num_threads) {
    int data_size = data.size();
    int chunk_size = data_size / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? data_size : (i + 1) * chunk_size;
        threads.emplace_back(extract_akaze_features, std::cref(data), std::ref(result), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
