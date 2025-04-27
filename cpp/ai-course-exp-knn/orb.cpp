#include "orb.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <mutex>

using namespace cv;

void extract_orb_features(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) {
    // 创建 ORB 特征检测器
    Ptr<ORB> orb = ORB::create();
    // 检测特征点并计算描述符
    orb->detectAndCompute(image, noArray(), keypoints, descriptors);
}
