#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <spdlog/spdlog.h>

#include "dataset.hpp"
#include "akaze.hpp"
#include "pca.hpp"

using namespace std;

void save_image(const vector<unsigned char>& images, int image_size, int index, const string& filename) {
    cv::Mat img(28, 28, CV_8UC1, (void*)&images[index * image_size]);
    cv::imwrite(filename, img);
}

#include <opencv2/opencv.hpp>

using namespace cv;

// 定义函数：使用双线性插值放大图像
void enlarge_image_linear(const Mat& inputImage, Mat& outputImage, double scaleFactor) {
    // 计算放大后的尺寸
    int newWidth = static_cast<int>(inputImage.cols * scaleFactor);
    int newHeight = static_cast<int>(inputImage.rows * scaleFactor);
    Size newSize(newWidth, newHeight);

    // 使用双线性插值放大图像
    resize(inputImage, outputImage, newSize, 0, 0, INTER_LINEAR);
}

// 定义函数：使用双三次插值放大图像
void enlarge_image_cubic(const Mat& inputImage, Mat& outputImage, double scaleFactor) {
    // 计算放大后的尺寸
    int newWidth = static_cast<int>(inputImage.cols * scaleFactor);
    int newHeight = static_cast<int>(inputImage.rows * scaleFactor);
    Size newSize(newWidth, newHeight);

    // 使用双三次插值放大图像
    resize(inputImage, outputImage, newSize, 0, 0, INTER_CUBIC);
}

void test_akaze() {
    spdlog::info("OpenCV version: {}", CV_VERSION);

    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");
    auto& train_data = fminist.get_train();

    // 定义存储特征点和描述符的变量
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // 放大
    cv::Mat enlargedImageLinear;
    enlarge_image_linear(train_data[0].first, enlargedImageLinear, 20.0);
    //enlargeImageLinear(cv::imread("E:\\repos\\playground\\cpp_playground\\ai-course-exp-knn\\a.jpg"), enlargedImageLinear, 20.0);

    // 提取特征点
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(enlargedImageLinear, noArray(), keypoints, descriptors);

    //cv::Mat test_img = cv::imread("E:\\repos\\playground\\cpp_playground\\ai-course-exp-knn\\a.jpg");
    //cv::Mat test_img = enlargedImageLinear;
    //extract_orb_features(test_img, keypoints, descriptors);

    // 输出提取到的特征点和描述符信息
    std::cout << "Number of keypoints: " << keypoints.size() << std::endl;
    std::cout << "Descriptors size: " << descriptors.rows << " x " << descriptors.cols << std::endl;

    // 在图像中绘制特征点
    cv::Mat output;
    cv::drawKeypoints(enlargedImageLinear, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // 显示结果
    cv::imshow("AKAZE Keypoints", output);
    cv::waitKey(0);
}

int main() {
    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");
    vector<cv::Mat> enlarged_train;
    //for(const auto& p : fminist.get_train()) {
    //    cv::Mat enlarged;
    //    enlarge_image_linear(p.first,enlarged,10.0f);
    //    enlarged_train.emplace_back(enlarged);
    //}
    auto& train_data = fminist.get_train();
    for(int i = 0;i < 100;i++) {
        cv::Mat enlarged;
        enlarge_image_linear(train_data[i].first,enlarged,10.0f);
        enlarged_train.emplace_back(enlarged);
    }

    std::vector<Mat> descriptors;
    extract_akaze_features(enlarged_train,descriptors,0,enlarged_train.size());

    cv::Mat pca_result = pca(descriptors,50);
    spdlog::info("pca_result size: {}, {}",pca_result.size[0],pca_result.size[1]);
    cv::imshow("pca_result",pca_result);
    cv::waitKey(0);
    return 0;
}