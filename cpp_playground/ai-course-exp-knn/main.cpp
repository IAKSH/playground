#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <spdlog/spdlog.h>

#include <opencv2/ml.hpp>

#include "dataset.hpp"
#include "akaze.hpp"
#include "pca.hpp"

using namespace std;
using namespace cv;

void save_image(const vector<unsigned char>& images, int image_size, int index, const string& filename) {
    cv::Mat img(28, 28, CV_8UC1, (void*)&images[index * image_size]);
    cv::imwrite(filename, img);
}

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

std::vector<cv::Mat> extract(const std::vector<std::pair<cv::Mat,int>>& data, double scale = 10.0, int n = 0) {
    vector<cv::Mat> enlarged_train;
    if(n == 0)
        n = data.size();
    for(int i = 0;i < n;i++) {
        cv::Mat enlarged;
        enlarge_image_linear(data[i].first, enlarged, scale);
        enlarged_train.emplace_back(enlarged);
    }

    std::vector<Mat> descriptors;
    //extract_akaze_features(enlarged_train,descriptors,0,enlarged_train.size());
    extract_akaze_features_mt(enlarged_train, descriptors, 8);

    return pca(descriptors,16);
}

static constexpr int TRAIN_LOAD_CNT = 10;
static constexpr int VAL_LOAD_CNT = 10;

int main() {
    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");

    auto& train_data = fminist.get_train();
    std::vector<cv::Mat> train_pca = extract(train_data,10.0,TRAIN_LOAD_CNT);
    cv::Mat train_labels(TRAIN_LOAD_CNT,1,CV_32F);
    for(int i = 0;i < TRAIN_LOAD_CNT;i++)
        train_labels.at<float>(i,1) = train_data[i].second;

    auto& val_data = fminist.get_val();
    std::vector<cv::Mat> val_pca = extract(val_data,10.0,VAL_LOAD_CNT);
    cv::Mat val_labels(VAL_LOAD_CNT,1,CV_32F);
    for(int i = 0;i < VAL_LOAD_CNT;i++)
        val_labels.at<float>(i,1) = val_data[i].second;

    // TODO: KNN

    return 0;
}