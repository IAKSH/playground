#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

void extract_akaze_features(const std::vector<cv::Mat>& mats, std::vector<cv::Mat>& keypoints_result,
    std::vector<cv::Mat>& descriptors_result, int start, int end);
void extract_akaze_features_mt(const std::vector<cv::Mat>& mats, std::vector<cv::Mat>& keypoints_result,
    std::vector<cv::Mat>& descriptors_result, int num_threads);