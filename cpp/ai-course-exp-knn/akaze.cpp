#include "akaze.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <mutex>

using namespace cv;

void extract_akaze_features(const std::vector<Mat>& mats, std::vector<Mat>& keypoints_result,
    std::vector<Mat>& descriptors_result, int start, int end) {
    for (int i = start; i < end; ++i) {
        Ptr<AKAZE> akaze = AKAZE::create();
        Mat descriptors;
        std::vector<KeyPoint> keypoints;
        akaze->detectAndCompute(mats[i], noArray(), keypoints, descriptors);

        // 将特征点转换为Mat
        Mat keypoints_mat(keypoints.size(), 7, CV_32F); // 每个特征点有7个属性
        for (size_t j = 0; j < keypoints.size(); ++j) {
            keypoints_mat.at<float>(j, 0) = keypoints[j].pt.x;
            keypoints_mat.at<float>(j, 1) = keypoints[j].pt.y;
            keypoints_mat.at<float>(j, 2) = keypoints[j].size;
            keypoints_mat.at<float>(j, 3) = keypoints[j].angle;
            keypoints_mat.at<float>(j, 4) = keypoints[j].response;
            keypoints_mat.at<float>(j, 5) = keypoints[j].octave;
            keypoints_mat.at<float>(j, 6) = keypoints[j].class_id;
        }

        keypoints_result[i] = keypoints_mat; // 将每个图像的特征点存储在结果中

        // 将描述子转换为CV_32F
        descriptors.convertTo(descriptors, CV_32F);
        descriptors_result[i] = descriptors; // 将每个图像的描述符存储在结果中
    }
}

static std::mutex descriptor_mutex;

static void extract_akaze_features_thread(const std::vector<cv::Mat>& mats, std::vector<Mat>& keypoints_result, std::vector<Mat>& descriptors_result, int start, int end) {
    for (int i = start; i < end; ++i) {
        Ptr<AKAZE> akaze = AKAZE::create();
        Mat descriptors;
        std::vector<KeyPoint> keypoints;
        akaze->detectAndCompute(mats[i], noArray(), keypoints, descriptors);

        // 将特征点转换为Mat
        Mat keypoints_mat(keypoints.size(), 7, CV_32F); // 每个特征点有7个属性
        for (size_t j = 0; j < keypoints.size(); ++j) {
            keypoints_mat.at<float>(j, 0) = keypoints[j].pt.x;
            keypoints_mat.at<float>(j, 1) = keypoints[j].pt.y;
            keypoints_mat.at<float>(j, 2) = keypoints[j].size;
            keypoints_mat.at<float>(j, 3) = keypoints[j].angle;
            keypoints_mat.at<float>(j, 4) = keypoints[j].response;
            keypoints_mat.at<float>(j, 5) = keypoints[j].octave;
            keypoints_mat.at<float>(j, 6) = keypoints[j].class_id;
        }

        {
            std::lock_guard<std::mutex> lock(descriptor_mutex);
            keypoints_result[i] = keypoints_mat; // 将每个图像的特征点存储在结果中
            descriptors.convertTo(descriptors, CV_32F); // 将描述子转换为CV_32F
            descriptors_result[i] = descriptors; // 将每个图像的描述符存储在结果中
        }
    }
}

void extract_akaze_features_mt(const std::vector<cv::Mat>& mats, std::vector<Mat>& keypoints_result,
    std::vector<Mat>& descriptors_result, int num_threads) {
    int data_size = mats.size();
    keypoints_result.resize(data_size); // 预先调整特征点结果向量的大小
    descriptors_result.resize(data_size); // 预先调整描述符结果向量的大小
    int chunk_size = data_size / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? data_size : (i + 1) * chunk_size;
        threads.emplace_back(extract_akaze_features_thread, std::cref(mats), std::ref(keypoints_result), std::ref(descriptors_result), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
