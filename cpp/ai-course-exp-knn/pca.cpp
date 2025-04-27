#include "pca.hpp"
#include <spdlog/spdlog.h>
#include <vector>
#include <thread>

cv::Mat pca(cv::Mat descriptors_list, int num_components) {
    // 检查输入是否为空
    if (descriptors_list.empty()) {
        spdlog::warn("The input descriptors list is empty. Returning zero-filled matrix.");
        // 返回一个大小合适的零填充矩阵
        return cv::Mat::zeros(1, num_components, CV_32F); // 假设返回1行num_components列的零填充矩阵
    }

    // 执行PCA操作
    cv::PCA pca(descriptors_list, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
    
    // 投影到主成分空间
    cv::Mat projected;
    pca.project(descriptors_list, projected);

    // 检查投影后的矩阵列数是否小于num_components
    if(projected.cols < num_components) {
        spdlog::warn("projected.size[1] = {} (num_components is {}), PCA miss", projected.cols, num_components);
        // 创建新的填充矩阵，初始化为零
        cv::Mat padded_projected = cv::Mat::zeros(projected.rows, num_components, projected.type());
        // 将已有的投影矩阵数据拷贝到填充矩阵中
        cv::Rect roi(0, 0, projected.cols, projected.rows);
        projected.copyTo(padded_projected(roi));
        return padded_projected;
    }

    return projected;
}

cv::Mat pca(const cv::Mat& descriptors_list, int num_components, int num_threads) {
    // 检查输入是否为空
    if (descriptors_list.empty()) {
        spdlog::warn("The input descriptors list is empty. Returning zero-filled matrix.");
        return cv::Mat::zeros(1, num_components, CV_32F);
    }

    // 切分数据为num_threads块
    std::vector<cv::Mat> data_blocks(num_threads);
    int block_size = descriptors_list.rows / num_threads;
    int remaining = descriptors_list.rows % num_threads;

    int start_row = 0;
    for (int i = 0; i < num_threads; ++i) {
        int end_row = start_row + block_size + (i < remaining ? 1 : 0);
        data_blocks[i] = descriptors_list.rowRange(start_row, end_row);
        start_row = end_row;
    }

    // 并行执行PCA
    std::vector<cv::Mat> projected_blocks(num_threads);
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            cv::PCA pca(data_blocks[i], cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
            pca.project(data_blocks[i], projected_blocks[i]);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // 合并结果
    cv::Mat projected;
    cv::vconcat(projected_blocks, projected);

    // 检查投影后的矩阵列数是否小于num_components
    if (projected.cols < num_components) {
        spdlog::warn("projected.size[1] = {} (num_components is {}), PCA miss", projected.cols, num_components);
        cv::Mat padded_projected = cv::Mat::zeros(projected.rows, num_components, projected.type());
        cv::Rect roi(0, 0, projected.cols, projected.rows);
        projected.copyTo(padded_projected(roi));
        return padded_projected;
    }

    return projected;
}
