#include "knn.hpp"

#define USE_MINKOWSKI_DISTANCE

#ifdef USE_MINKOWSKI_DISTANCE
static double minkowski_distance(const cv::Mat& a, const cv::Mat& b, double p) {
    double sum = 0.0;
    for (int i = 0; i < a.cols; ++i) {
        double diff = std::abs(a.at<float>(0, i) - b.at<float>(0, i));
        sum += std::pow(diff, p);
    }
    return std::pow(sum, 1.0 / p);
}
#else
static double distance(const cv::Mat& a, const cv::Mat& b) {
    double sum = 0.0;
    for (int i = 0; i < a.cols; ++i) {
        double diff = a.at<float>(0, i) - b.at<float>(0, i);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
#endif

void knn_predict(const cv::Mat& train_features, const cv::Mat& train_labels, const cv::Mat& sample, int k, int& result) {
    std::vector<std::pair<double, int>> distances;

    for (int i = 0; i < train_features.rows; ++i) {
#ifdef USE_MINKOWSKI_DISTANCE
        double dist = minkowski_distance(train_features.row(i), sample, 3.0);
#else
        double dist = distance(train_features.row(i), sample);
#endif
        distances.push_back(std::make_pair(dist, train_labels.at<int>(i, 0)));
    }

    std::sort(distances.begin(), distances.end());
    std::vector<int> neighbors;

    for (int i = 0; i < k; ++i) {
        neighbors.push_back(distances[i].second);
    }

    std::map<int, int> counts;
    for (int label : neighbors) {
        counts[label]++;
    }

    result = std::max_element(counts.begin(), counts.end(), [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}

void knn_validate(const cv::Mat& train_features, const cv::Mat& train_labels, const cv::Mat& val_features, const cv::Mat& val_labels, int k, int start, int end, int& correct) {
    for (int i = start; i < end; ++i) {
        int predicted_label;
        knn_predict(train_features, train_labels, val_features.row(i), k, predicted_label);
        if (predicted_label == val_labels.at<int>(i, 0)) {
            correct++;
        }
    }
}