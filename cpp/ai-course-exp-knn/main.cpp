#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <cstdint>

#include <spdlog/spdlog.h>
#include <opencv2/ml.hpp>

#include "dataset.hpp"
#include "pca.hpp"
#include "knn.hpp"

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

    FashionMINIST fminist("../dataset/FashionMNIST");
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

std::pair<cv::Mat, cv::Mat> extract(const std::vector<std::pair<cv::Mat, int>>& data, double scale = 10.0, int n = 0) {
    n = (n == 0 ? data.size() : n);

    spdlog::info("collecting labels");

    // 获取label
    cv::Mat labels(n, 1, CV_32F);
    for (int i = 0; i < n; i++)
        labels.at<float>(i, 0) = data[i].second;

    spdlog::info("enlarging images");

    // 放大图片
    std::vector<cv::Mat> enlarged_images;
    for (int i = 0; i < n; i++) {
        cv::Mat enlarged;
        enlarge_image_cubic(data[i].first, enlarged, scale);
        enlarged_images.emplace_back(enlarged);
    }

    spdlog::info("converting images to vectors");

    // 将每张图像转换为一个行向量，并存储到feature_vectors中
    std::vector<cv::Mat> feature_vectors;
    int max_cols = 0;

    for (const auto& img : enlarged_images) {
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F);
        cv::Mat feature_vector = img_float.reshape(1, 1); // 将图像转换为单行矩阵

        if (feature_vector.cols > max_cols) {
            max_cols = feature_vector.cols;
        }

        feature_vectors.push_back(feature_vector);
    }

    spdlog::info("filling zero");

    // 填充零使得所有特征向量的列数一致
    for (auto& vec : feature_vectors) {
        if (vec.cols < max_cols) {
            cv::copyMakeBorder(vec, vec, 0, 0, 0, max_cols - vec.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }

    spdlog::info("concatenating");

    // 将所有特征向量拼接成一个矩阵
    cv::Mat all_feature_vectors;
    cv::vconcat(feature_vectors, all_feature_vectors);

    spdlog::info("doing PCA");

    // 最终PCA降维，得到最终特征向量
    cv::Mat final_features = pca(all_feature_vectors, 128, 4);

    spdlog::info("extract done");

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

//#define TRAINNING
static constexpr int TRAIN_LOAD_CNT = 10000;
static constexpr int VAL_LOAD_CNT = 1000;

void extract_fashion_minist() {
    spdlog::info("loading FashionMinist");
    FashionMINIST fminist("../dataset/FashionMNIST");
    spdlog::info("extracting");
    auto& train_data = fminist.get_train();
    auto& val_data = fminist.get_val();
    auto train = extract(train_data,10.0,TRAIN_LOAD_CNT);
    auto val = extract(val_data,10.0,VAL_LOAD_CNT);

    spdlog::info("train_feature_size: {},{}",train.first.size[0],train.first.size[1]);
    spdlog::info("train_label_size: {},{}",train.second.size[0],train.second.size[1]);

    spdlog::info("saving");
    cv::FileStorage file("data.yml", cv::FileStorage::WRITE);
    file << "train_features" << train.first;
    file << "train_labels" << train.second;
    file << "val_features" << val.first;
    file << "val_labels" << val.second;
    file.release();
    spdlog::info("done");
}

// FashionMNIST labels
const std::vector<std::string> fashion_mnist_labels = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

void predict_and_display(int n, int k) {
    spdlog::info("loading data.yml");
    std::pair<cv::Mat, cv::Mat> train;
    cv::FileStorage file_read("data.yml", cv::FileStorage::READ);
    file_read["train_features"] >> train.first;
    file_read["train_labels"] >> train.second;
    file_read.release();

    if (n < 0 || n >= train.first.rows) {
        spdlog::error("Index n is out of range");
        return;
    }

    cv::Mat sample = train.first.row(n);
    int predicted_label;
    knn_predict(train.first, train.second, sample, k, predicted_label);

    spdlog::info("Predicted label for sample {}: {}", n, fashion_mnist_labels[predicted_label]);

    spdlog::info("loading FashionMinist");
    FashionMINIST fminist("../dataset/FashionMNIST");

    cv::Mat image = fminist.get_val()[n].first;
    cv::imshow("Sample Image", image);
    cv::waitKey(0); // Wait for key press
}

void test_knn(int k, int validation_size) {
    spdlog::info("loading data.yml");
    std::pair<cv::Mat,cv::Mat> train, val;
    cv::FileStorage file_read("data.yml", cv::FileStorage::READ);
    file_read["train_features"] >> train.first;
    file_read["train_labels"] >> train.second;
    file_read["val_features"] >> val.first;
    file_read["val_labels"] >> val.second;
    file_read.release();

    validation_size = std::min(validation_size, val.first.rows);
    cv::Mat val_features = val.first.rowRange(0, validation_size);
    cv::Mat val_labels = val.second.rowRange(0, validation_size);

    spdlog::info("running multi-thread validation");

    int correct = 0;
    int num_threads = 8; 
    std::vector<std::thread> threads;
    int step = val_features.rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start = i * step;
        int end = (i == num_threads - 1) ? val_features.rows : (i + 1) * step;
        threads.emplace_back(knn_validate, std::ref(train.first), std::ref(train.second), std::ref(val_features), std::ref(val_labels), k, start, end, std::ref(correct));
    }

    for (auto& t : threads) {
        t.join();
    }

    float accuracy = correct / static_cast<float>(val_features.rows);
    
    spdlog::info("total: {}", val_features.rows);
    spdlog::info("correct: {}", correct);
    spdlog::info("accuracy: {}%", accuracy * 100.0);
}

void find_best_k(int min_k, int max_k, int validation_size) {
    spdlog::info("loading data.yml");
    std::pair<cv::Mat, cv::Mat> train, val;
    cv::FileStorage file_read("data.yml", cv::FileStorage::READ);
    file_read["train_features"] >> train.first;
    file_read["train_labels"] >> train.second;
    file_read["val_features"] >> val.first;
    file_read["val_labels"] >> val.second;
    file_read.release();

    validation_size = std::min(validation_size, val.first.rows);
    cv::Mat val_features = val.first.rowRange(0, validation_size);
    cv::Mat val_labels = val.second.rowRange(0, validation_size);

    spdlog::info("running multi-thread validation");

    int best_k = min_k;
    float best_accuracy = 0.0;

    for (int k = min_k; k <= max_k; ++k) {
        int correct = 0;
        int num_threads = 8;
        std::vector<std::thread> threads;
        int step = val_features.rows / num_threads;

        for (int i = 0; i < num_threads; i++) {
            int start = i * step;
            int end = (i == num_threads - 1) ? val_features.rows : (i + 1) * step;
            threads.emplace_back(knn_validate, std::ref(train.first), std::ref(train.second), std::ref(val_features), std::ref(val_labels), k, start, end, std::ref(correct));
        }

        for (auto& t : threads) {
            t.join();
        }

        float accuracy = correct / static_cast<float>(val_features.rows);
        
        spdlog::info("k: {}, total: {}, correct: {}, accuracy: {}%", k, val_features.rows, correct, accuracy * 100.0);

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_k = k;
        }
    }

    spdlog::info("best k: {}, best accuracy: {}%", best_k, best_accuracy * 100.0);
}

void save_train_image(int n) {
    FashionMINIST fminist("../dataset/FashionMNIST");
    auto& train_data = fminist.get_train();
    for(int i = 0;i < n;i++)
        cv::imwrite(std::format("train_images/train_{}.jpg",i),train_data[i].first);
}

int main() {
#ifdef TRAINNING
    extract_fashion_minist();
#else
    //test_knn(150, VAL_LOAD_CNT);
    //find_best_k(130,140,VAL_LOAD_CNT);
    predict_and_display(1,132);
#endif
    return 0;
}