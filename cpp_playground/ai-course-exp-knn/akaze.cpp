#include "akaze.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <mutex>

using namespace cv;

void extract_akaze_features(const std::vector<Mat>& mats, std::vector<Mat>& result, int start, int end) {
    for (int i = start; i < end; ++i) {
        Ptr<AKAZE> akaze = AKAZE::create();
        Mat descriptors;
        std::vector<KeyPoint> keypoints;
        akaze->detectAndCompute(mats[i], noArray(), keypoints, descriptors);
        result.push_back(descriptors);
    }
}

static std::mutex descriptor_mutex;

static void extract_akaze_features_thread(const std::vector<cv::Mat>& mats, std::vector<Mat>& result, int start, int end) {
    for (int i = start; i < end; ++i) {
        Ptr<AKAZE> akaze = AKAZE::create();
        Mat descriptors;
        std::vector<KeyPoint> keypoints;

        akaze->detectAndCompute(mats[i], noArray(), keypoints, descriptors);

        std::lock_guard<std::mutex> lock(descriptor_mutex);
        result.push_back(descriptors);
    }
}

void extract_akaze_features_mt(const std::vector<cv::Mat>& mats, std::vector<Mat>& result, int num_threads) {
    int data_size = mats.size();
    int chunk_size = data_size / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? data_size : (i + 1) * chunk_size;
        threads.emplace_back(extract_akaze_features_thread, std::cref(mats), std::ref(result), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
