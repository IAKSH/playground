#include "pca.hpp"

cv::Mat pca(const std::vector<cv::Mat>& descriptors_list, int num_components) {
    // 确定最大尺寸
    int max_rows = 0;
    int max_cols = 0;
    for (const auto& desc : descriptors_list) {
        if (desc.rows > max_rows) max_rows = desc.rows;
        if (desc.cols > max_cols) max_cols = desc.cols;
    }

    // 调整大小并合并数据
    cv::Mat data;
    for (const auto& desc : descriptors_list) {
        cv::Mat resized_desc;
        cv::resize(desc, resized_desc, cv::Size(max_cols, max_rows));
        data.push_back(resized_desc.reshape(1, 1)); // 将每个矩阵转换为一行
    }

    // 执行PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);

    // 获取降维后的特征
    cv::Mat projected_data = pca.project(data);
    return projected_data;
}