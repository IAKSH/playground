#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
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

std::pair<cv::Mat,cv::Mat> extract(const std::vector<std::pair<cv::Mat,int>>& data, double scale = 10.0, int n = 0) {
    n = (n == 0 ? data.size() : n);

    // 获取label
    cv::Mat labels(n, 1, CV_32F);
    for(int i = 0; i < n; i++)
        labels.at<float>(i, 0) = data[i].second;

    // 放大图片
    std::vector<cv::Mat> enlarged_images;
    for(int i = 0;i < n;i++) {
        cv::Mat enlarged;
        enlarge_image_linear(data[i].first, enlarged, scale);
        enlarged_images.emplace_back(enlarged);
    }

    // 提取特征点
    std::vector<std::vector<cv::Mat>> descriptors_per_item(n);
    extract_akaze_features_mt(enlarged_images, descriptors_per_item, 8); 

    // 对每个项进行初步的PCA降维，然后拼接特征点，再进行最终的PCA降维
    std::vector<cv::Mat> reduced_descriptors_list;
    for (int i = 0; i < n; i++) {
        cv::Mat concatenated_descriptors;

        cv::vconcat(descriptors_per_item[i], concatenated_descriptors);
        cv::Mat reduced_descriptors = pca(concatenated_descriptors, 32); // 初步降维

        if(reduced_descriptors.size[1] != 32)
            spdlog::warn("reduced_descriptors.size[1] = {}, concatenated_descriptors.size[1] = {}",i,reduced_descriptors.size[1],concatenated_descriptors.size[1]);

        reduced_descriptors_list.push_back(reduced_descriptors);
    }

    // 最终PCA降维，得到最终特征向量
    cv::Mat final_descriptors;
    cv::vconcat(reduced_descriptors_list, final_descriptors);
    cv::Mat final_features = pca(final_descriptors, 16);

    return std::make_pair(final_features, labels);
}

static mutex mtx;

void validate(cv::Ptr<cv::ml::KNearest> knn, cv::Mat& val_first, cv::Mat& val_second, int start, int end, int& correct) {
    for (int i = start; i < end; i++) {
        cv::Mat response, dists;
        float prediction = knn->findNearest(val_first.row(i), 3, response, dists);
        if (prediction == val_second.at<float>(i, 0)) {
            lock_guard<mutex> lock(mtx);
            correct++;
        }
    }
}

static constexpr int TRAIN_LOAD_CNT = 100;
static constexpr int VAL_LOAD_CNT = 10;

void extract_fashion_minist() {
    spdlog::info("loading FashionMinist");
    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");
    spdlog::info("extracting");
    auto& train_data = fminist.get_train();
    auto& val_data = fminist.get_val();
    auto train = extract(train_data,10.0,TRAIN_LOAD_CNT);
    auto val = extract(val_data,10.0,VAL_LOAD_CNT);

    spdlog::info("saving");
    cv::FileStorage file("data.yml", cv::FileStorage::WRITE);
    file << "train_features" << train.first;
    file << "train_labels" << train.second;
    file << "val_features" << val.first;
    file << "val_labels" << val.second;
    file.release();
    spdlog::info("done");
}

void test_knn() {
    std::pair<cv::Mat,cv::Mat> train,val;
    cv::FileStorage fileRead("data.yml", cv::FileStorage::READ);
    fileRead["train_features"] >> train.first;
    fileRead["train_labels"] >> train.second;
    fileRead["val_features"] >> val.first;
    fileRead["val_labels"] >> val.second;
    fileRead.release();

    // KNN
    spdlog::info("training knn");
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->train(train.first, cv::ml::ROW_SAMPLE, train.second);

    spdlog::info("running multi-thread validation");
    // Test accuracy with multithreading
    int correct = 0;
    int num_threads = 8; // Adjust the number of threads as needed
    vector<thread> threads;
    int step = val.first.size[0] / num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start = i * step;
        int end = (i == num_threads - 1) ? val.first.size[0] : (i + 1) * step;
        threads.emplace_back(validate, knn, std::ref(val.first), std::ref(val.second), start, end, std::ref(correct));
    }

    for (auto& t : threads) {
        t.join();
    }

    float accuracy = correct / static_cast<float>(val.first.size[0]);
    cout << "Accuracy: " << accuracy * 100.0 << "%" << endl;
}

int main() {
    extract_fashion_minist();
    return 0;
}