#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

void extract_akaze_features(const std::vector<std::pair<cv::Mat, int>>& data, std::vector<cv::Mat>& result, int start, int end);
void extract_akaze_features_mt(const std::vector<std::pair<cv::Mat, int>>& data, std::vector<cv::Mat>& result, int num_threads);