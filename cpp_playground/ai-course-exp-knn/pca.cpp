#include "pca.hpp"
#include <spdlog/spdlog.h>

/*
cv::Mat pca(const std::vector<cv::Mat>& descriptors_list, int num_components) {
    // descriptors_list中的每一项大小为[n,61]，n表示有多少个特征点，可能为0
    // 对于没有特征点的项，依然要输出一个对应的pca结果行
    // ...
}
*/

std::vector<cv::Mat> pca(const std::vector<cv::Mat>& descriptors_list, int num_components) {
    // Collect all descriptors into one large matrix
    cv::Mat all_descriptors;
    for (const auto& descriptors : descriptors_list) {
        if (!descriptors.empty()) {
            all_descriptors.push_back(descriptors);
        }
    }

    // Perform PCA on the large matrix
    cv::PCA pca(all_descriptors, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
    
    // Project each descriptor list into the PCA space
    std::vector<cv::Mat> pca_results;
    for (const auto& descriptors : descriptors_list) {
        cv::Mat pca_result;
        if (!descriptors.empty()) {
            pca_result = pca.project(descriptors);
        } else {
            // Handle empty descriptors by providing a zero matrix with the required components
            pca_result = cv::Mat::zeros(1, num_components, CV_32F);
        }
        pca_results.push_back(pca_result);
    }

    // Convert the list of pca_results back to a single matrix if needed
    return pca_results;
}
