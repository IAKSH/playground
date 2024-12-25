#include "pca.hpp"
#include <spdlog/spdlog.h>

cv::Mat pca(cv::Mat descriptors_list, int num_components) {
    // 检查输入是否为空
    if (descriptors_list.empty())
        throw std::invalid_argument("The input descriptors list is empty.");

    // 执行PCA操作
    cv::PCA pca(descriptors_list, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
    
    // 投影到主成分空间
    cv::Mat projected;
    pca.project(descriptors_list, projected);

    // 检查投影后的矩阵列数是否小于num_components
    if(projected.cols < num_components) {
        spdlog::warn("projected.size[1] = {} (num_components is {}), PCA miss",projected.cols,num_components);
        // 创建新的填充矩阵，初始化为零
        cv::Mat padded_projected = cv::Mat::zeros(projected.rows, num_components, projected.type());
        // 将已有的投影矩阵数据拷贝到填充矩阵中
        cv::Rect roi(0, 0, projected.cols, projected.rows);
        projected.copyTo(padded_projected(roi));
        return padded_projected;
    }

    return projected;
}
