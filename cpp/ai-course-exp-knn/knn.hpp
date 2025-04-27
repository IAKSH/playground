#pragma once
#include <opencv2/opencv.hpp>

void knn_validate(const cv::Mat& train_features, const cv::Mat& train_labels,
    const cv::Mat& val_features, const cv::Mat& val_labels, int k, int start, int end, int& correct);

void knn_predict(const cv::Mat& train_features, const cv::Mat& train_labels,
    const cv::Mat& sample, int k, int& result);