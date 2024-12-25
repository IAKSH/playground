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

std::pair<cv::Mat, cv::Mat> extract(const std::vector<std::pair<cv::Mat, int>>& data, double scale = 10.0, int n = 0) {
    n = (n == 0 ? data.size() : n);

    // 获取label
    cv::Mat labels(n, 1, CV_32F);
    for (int i = 0; i < n; i++)
        labels.at<float>(i, 0) = data[i].second;

    // 放大图片
    std::vector<cv::Mat> enlarged_images;
    for (int i = 0; i < n; i++) {
        cv::Mat enlarged;
        enlarge_image_cubic(data[i].first, enlarged, scale);
        enlarged_images.emplace_back(enlarged);
    }

    // 提取特征点
    std::vector<std::vector<cv::Mat>> descriptors_per_item(n);
    extract_akaze_features_mt(enlarged_images, descriptors_per_item, 8);

    // 从每个item的描述子中提取特征向量
    cv::Mat feature_vectors(n, 61, CV_32F); // 假设每个描述子的长度为61
    for (int i = 0; i < n; i++) {
        bool b = true;
        for(const auto& mat : descriptors_per_item[i]) {
            if(mat.empty()) {
                b = false;
                break;
            }
        }
        if (b && !descriptors_per_item[i].empty()) {
            cv::Mat descriptors;
            cv::vconcat(descriptors_per_item[i], descriptors);
            cv::reduce(descriptors, feature_vectors.row(i), 0, cv::REDUCE_SUM2);
        } else {
            spdlog::warn("empty descriptor, using zeros instead");
            feature_vectors.row(i) = cv::Mat::zeros(1, 61, CV_32F); // 若无描述子，则使用零向量
        }
    }

    // 最终PCA降维，得到最终特征向量
    cv::Mat final_features = pca(feature_vectors, 32);

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

static constexpr int TRAIN_LOAD_CNT = 0;
static constexpr int VAL_LOAD_CNT = 0;

void extract_fashion_minist() {
    spdlog::info("loading FashionMinist");
    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");
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

void predict_and_display(const cv::Ptr<cv::ml::KNearest>& knn, const cv::Mat& val_features, int n) {
    // 确保n在有效范围内
    if (n < 0 || n >= val_features.rows) {
        spdlog::error("Index out of range");
        return;
    }

    // 对验证集中的第n个图片进行预测
    cv::Mat sample = val_features.row(n);
    cv::Mat result;
    knn->findNearest(sample, knn->getDefaultK(), result);
    int predicted_label = static_cast<int>(result.at<float>(0, 0));

    // 定义FashionMNIST的标签字符描述
    std::vector<std::string> fashion_mnist_labels = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    // 输出预测结果对应的类型描述
    if (predicted_label >= 0 && predicted_label < fashion_mnist_labels.size()) {
        spdlog::info("Predicted label: {}", fashion_mnist_labels[predicted_label]);
    } else {
        spdlog::error("Predicted label out of range!");
    }

    // 显示验证集中的第n个图片
    FashionMINIST fminist("E:\\repos\\playground\\python_playground\\ai-course-exp-knn\\data\\FashionMNIST");
    auto& val_data = fminist.get_val();

    cv::Mat image = val_data[n].first;
    cv::imshow("Validation Image", image);
    cv::waitKey(0);
}

void test_knn(int k, int validation_size) {
    spdlog::info("loading data.yml");
    std::pair<cv::Mat,cv::Mat> train, val;
    cv::FileStorage fileRead("data.yml", cv::FileStorage::READ);
    fileRead["train_features"] >> train.first;
    fileRead["train_labels"] >> train.second;
    fileRead["val_features"] >> val.first;
    fileRead["val_labels"] >> val.second;
    fileRead.release();

    // 调整验证集数据量
    validation_size = std::min(validation_size, val.first.rows);
    cv::Mat val_features = val.first.rowRange(0, validation_size);
    cv::Mat val_labels = val.second.rowRange(0, validation_size);

    // KNN
    spdlog::info("training knn");
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(k);  // 设置k值
    knn->train(train.first, cv::ml::ROW_SAMPLE, train.second);

    spdlog::info("running multi-thread validation");
    // Test accuracy with multithreading
    int correct = 0;
    int num_threads = 8; // Adjust the number of threads as needed
    vector<thread> threads;
    int step = val_features.rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start = i * step;
        int end = (i == num_threads - 1) ? val_features.rows : (i + 1) * step;
        threads.emplace_back(validate, knn, std::ref(val_features), std::ref(val_labels), start, end, std::ref(correct));
    }

    for (auto& t : threads) {
        t.join();
    }

    float accuracy = correct / static_cast<float>(val_features.rows);
    
    spdlog::info("total: {}", val_features.rows);
    spdlog::info("correct: {}", correct);
    spdlog::info("accuracy: {}%", accuracy * 100.0);

    // 选择验证集中的第n个图片进行预测并显示
    //predict_and_display(knn, val_features, 4);
}

int main() {
    //extract_fashion_minist();
    test_knn(1,10000);
    return 0;
}