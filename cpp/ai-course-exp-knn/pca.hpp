#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

cv::Mat pca(cv::Mat descriptors_list, int num_components);
cv::Mat pca(const cv::Mat& descriptors_list, int num_components, int num_threads);