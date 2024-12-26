#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <limits>

struct Point {
    double x, y;
    int label;
};

// 用于生成随机数据集并分配标签
std::vector<Point> generate_labeled_data(size_t num_points_per_region) {
    std::vector<Point> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 0.5);

    // 四个区域的标签
    std::vector<int> labels = {0, 1, 2, 3};
    std::vector<std::pair<double, double>> offsets = {{0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}, {0.5, 0.5}};

    for (size_t region = 0; region < 4; ++region) {
        for (size_t i = 0; i < num_points_per_region; ++i) {
            double x = offsets[region].first + dis(gen);
            double y = offsets[region].second + dis(gen);
            data.push_back({x, y, labels[region]});
        }
    }

    return data;
}

// 计算两个点之间的欧几里得距离
double euclidean_distance(const Point& p1, const Point& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// k-means聚类算法
std::vector<int> kmeans(const std::vector<Point>& data, size_t k, size_t max_iters) {
    size_t n = data.size();
    std::vector<Point> centroids(k);
    std::vector<int> labels(n, -1);
    std::random_device rd;
    std::mt19937 gen(rd());

    // 随机初始化质心
    for (size_t i = 0; i < k; ++i) {
        centroids[i] = data[std::uniform_int_distribution<>(0, n - 1)(gen)];
    }

    for (size_t iter = 0; iter < max_iters; ++iter) {
        // 分配每个点到最近的质心
        for (size_t i = 0; i < n; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < k; ++j) {
                double dist = euclidean_distance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    labels[i] = j;
                }
            }
        }

        // 更新质心位置
        std::vector<Point> new_centroids(k, {0.0, 0.0, 0});
        std::vector<size_t> counts(k, 0);
        for (size_t i = 0; i < n; ++i) {
            new_centroids[labels[i]].x += data[i].x;
            new_centroids[labels[i]].y += data[i].y;
            counts[labels[i]]++;
        }

        for (size_t j = 0; j < k; ++j) {
            if (counts[j] != 0) {
                new_centroids[j].x /= counts[j];
                new_centroids[j].y /= counts[j];
            }
        }

        centroids = new_centroids;
    }

    return labels;
}

// 计算聚类内误差平方和（WSS）
double calculate_wss(const std::vector<Point>& data, const std::vector<int>& labels, const std::vector<Point>& centroids) {
    double wss = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        wss += euclidean_distance(data[i], centroids[labels[i]]);
    }
    return wss;
}

// 计算调整兰德指数
double adjusted_rand_index(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    size_t n = true_labels.size();
    size_t a = 0, b = 0, c = 0, d = 0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            bool same_true = (true_labels[i] == true_labels[j]);
            bool same_pred = (predicted_labels[i] == predicted_labels[j]);

            if (same_true && same_pred) {
                a++;
            } else if (!same_true && !same_pred) {
                b++;
            } else if (same_true && !same_pred) {
                c++;
            } else {
                d++;
            }
        }
    }

    double rand_index = static_cast<double>(a + b) / (a + b + c + d);
    double expected_index = static_cast<double>(a + c) * (a + d) + static_cast<double>(b + d) * (b + c);
    expected_index /= (a + b + c + d) * (a + b + c + d);

    return (rand_index - expected_index) / (1.0 - expected_index);
}

int main() {
    // 初始化spdlog
    auto logger = spdlog::basic_logger_mt("basic_logger", "logs/kmeans.log");
    spdlog::set_level(spdlog::level::info); // 设置日志等级为info

    // 生成随机数据集和标签
    size_t num_points_per_region = 25;
    auto data = generate_labeled_data(num_points_per_region);
    std::vector<int> true_labels;
    for (const auto& point : data) {
        true_labels.push_back(point.label);
    }
    logger->info("Labeled data generated.");

    // 寻找最佳k值
    size_t max_k = 10;
    size_t max_iters = 100;
    double best_wss = std::numeric_limits<double>::max();
    size_t best_k = 1;
    std::vector<int> best_labels;
    std::vector<Point> centroids;

    for (size_t k = 1; k <= max_k; ++k) {
        auto labels = kmeans(data, k, max_iters);
        double wss = calculate_wss(data, labels, data);
        logger->info("k:{} wss: {}", k, wss);
        if (wss < best_wss) {
            best_wss = wss;
            best_k = k;
            best_labels = labels;

            // 更新质心
            centroids = std::vector<Point>(k, {0.0, 0.0, 0});
            std::vector<size_t> counts(k, 0);
            for (size_t i = 0; i < data.size(); ++i) {
                centroids[labels[i]].x += data[i].x;
                centroids[labels[i]].y += data[i].y;
                counts[labels[i]]++;
            }
            for (size_t j = 0; j < k; ++j) {
                if (counts[j] != 0) {
                    centroids[j].x /= counts[j];
                    centroids[j].y /= counts[j];
                }
            }
        }
    }
    logger->info("Best k found: {}", best_k);

    // 计算调整兰德指数
    double ari = adjusted_rand_index(true_labels, best_labels);
    logger->info("Adjusted Rand Index calculated: {}", ari);

    // 打印结果
    for (size_t i = 0; i < num_points_per_region * 4; ++i) {
        std::cout << "Point: (" << data[i].x << ", " << data[i].y << "), True Cluster: " << true_labels[i]
                  << ", Predicted Cluster: " << best_labels[i] << std::endl;
    }
    std::cout << "Adjusted Rand Index: " << ari << std::endl;

    // 从标准输入中获取一个坐标
    double input_x, input_y;
    std::cout << "Enter a coordinate (x y): ";
    std::cin >> input_x >> input_y;

    // 预测输入坐标的聚类结果
    Point input_point = {input_x, input_y, 0};
    double min_dist = std::numeric_limits<double>::max();
    int cluster = -1;
    for (size_t j = 0; j < best_k; ++j) {
        double dist = euclidean_distance(input_point, centroids[j]);
        if (dist < min_dist) {
            min_dist = dist;
            cluster = j;
        }
    }
    std::cout << "The input coordinate belongs to cluster: " << cluster << std::endl;

    return 0;
}
