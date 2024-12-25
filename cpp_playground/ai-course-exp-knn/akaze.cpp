#include "akaze.hpp"
#include <spdlog/spdlog.h>

using namespace cv;

void extract_akaze_features(const std::vector<std::pair<Mat, int>>& data, std::vector<Mat>& result, int start, int end) {
    for (int i = start; i < end; ++i) {
        Ptr<AKAZE> akaze = AKAZE::create();
        Mat descriptors;
        std::vector<KeyPoint> keypoints;
        akaze->detectAndCompute(data[i].first, noArray(), keypoints, descriptors);
        result.push_back(descriptors);

        if(descriptors.empty())
            spdlog::error("descriptors empty");
    }
}

void extract_akaze_features(const std::vector<Mat>& mats, std::vector<Mat>& result, int start, int end) {
    for (int i = start; i < end; ++i) {
        Ptr<AKAZE> akaze = AKAZE::create();
        Mat descriptors;
        std::vector<KeyPoint> keypoints;
        akaze->detectAndCompute(mats[i], noArray(), keypoints, descriptors);
        result.push_back(descriptors);
    }
}