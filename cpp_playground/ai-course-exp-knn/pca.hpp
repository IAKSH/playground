#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

std::vector<cv::Mat> pca(const std::vector<cv::Mat>& descriptors_list, int num_components);