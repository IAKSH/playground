#include "pca.hpp"

cv::Mat pca(const std::vector<cv::Mat>& descriptors_list, int num_components) {
    // 将特征向量平铺到一个矩阵中
    cv::Mat all_descriptors;
    for (const auto& desc : descriptors_list) {
        if (!desc.empty()) {
            // 确保所有描述符的大小一致
            cv::Mat desc_flat = desc.reshape(1, 1); 
            all_descriptors.push_back(desc_flat);
        }
    }

    // 进行 PCA 降维
    cv::PCA pca(all_descriptors, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
    cv::Mat reduced_descriptors;
    pca.project(all_descriptors, reduced_descriptors);

    return reduced_descriptors;
}