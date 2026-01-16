#include <stampit/image.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// TODO: 自动修补破损，优化图像边缘

namespace stampit {
    // 比较神奇的是，由于末尾的括号，这其实已经是一个对lambda的函数调用了，这个lambda只会执行一次
    // 带初始化逻辑的静态变量
    auto logger = []{
        auto lg = spdlog::stdout_color_mt("stampit");
        lg->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
        return lg;
    }();
}

stampit::Image::Image(std::string_view path)
    : mat{cv::imread(path.data())} {}

stampit::Image::Image(const Image& image)
    : mat{image.mat} {}

stampit::Image::Image(cv::Mat mat)
    : mat{mat} {}

stampit::Image stampit::Image::copy() {
    return Image(mat.clone());
}

void stampit::Image::denoise(int min_area) {
    if(mat.type() != CV_8UC1) {
        logger->error("trying to denoise an image that has no been binarized");
    }

    // 反转，把黑色区域当作前景
    cv::bitwise_not(mat,mat);

    // 连通域分析
    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(mat, labels, stats, centroids);

    mat = cv::Mat::zeros(mat.size(), CV_8UC1);
    for (int i = 1; i < n_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_area) {
            cv::Mat mask = (labels == i);
            mat.setTo(255, mask);
        }
    }

    // 再反转，恢复原始前景/背景语义
    cv::bitwise_not(mat, mat);
}

stampit::Rect stampit::Image::get_min_boundary(int margin) const {
    if(mat.type() != CV_8UC1) {
        logger->error("trying to detect boundary from an image that has no been binarized");
    }

    // 原始尺寸
    int orig_w = mat.cols;
    int orig_h = mat.rows;

    // 按比例缩放到宽度=256
    int target_w = 256;
    double scale = (double)target_w / orig_w;
    int target_h = cvRound(orig_h * scale);

    cv::Mat small;
    cv::resize(mat, small, cv::Size(target_w, target_h), 0, 0, cv::INTER_NEAREST);

    // 在缩小图上做处理（连通域 + minEnclosingCircle）
    cv::Mat inverted;
    cv::bitwise_not(small, inverted);

    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(inverted, labels, stats, centroids);
    if (n_labels <= 1)
        return Rect{0,0,0,0};

    cv::Point2f img_center(target_w / 2.0f, target_h / 2.0f);
    double max_area = 0;
    int best_idx = -1;

    for (int i = 1; i < n_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < 100) continue; // 面积阈值

        cv::Point2f centroid(centroids.at<double>(i,0), centroids.at<double>(i,1));
        double dist = cv::norm(centroid - img_center);

        if (dist < std::max(target_w, target_h) * 0.25) {
            if (area > max_area) {
                max_area = area;
                best_idx = i;
            }
        }
    }

    if (best_idx == -1)
        return Rect(0,0,0,0);

    std::vector<cv::Point> points;
    for (int y = 0; y < labels.rows; ++y) {
        const int* row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] == best_idx) {
                points.emplace_back(x, y);
            }
        }
    }

    if (points.empty())
        return Rect();

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(points, center, radius);

    cv::Rect rect_small(cvRound(center.x - radius),
                       cvRound(center.y - radius),
                       cvRound(radius * 2),
                       cvRound(radius * 2));

    rect_small &= cv::Rect(0, 0, target_w, target_h);

    // 映射回原始尺寸
    cv::Rect rect_orig(cvRound(rect_small.x / scale),
                      cvRound(rect_small.y / scale),
                      cvRound(rect_small.width / scale),
                      cvRound(rect_small.height / scale));

    rect_orig &= cv::Rect(0, 0, orig_w, orig_h);

    if(margin) {
        // 在原始图像坐标系里扩展矩形
        rect_orig.x = std::max(0, rect_orig.x - margin);
        rect_orig.y = std::max(0, rect_orig.y - margin);
        rect_orig.width  = std::min(mat.cols - rect_orig.x, rect_orig.width  + 2 * margin);
        rect_orig.height = std::min(mat.rows - rect_orig.y, rect_orig.height + 2 * margin);
    }

    return Rect{rect_orig.x,rect_orig.y,rect_orig.width,rect_orig.height};
}

void stampit::Image::threshold(uint8_t th) {
    // 转灰度
    cv::cvtColor(mat, mat, cv::COLOR_BGRA2GRAY);
    // 二值化
    cv::threshold(mat,mat,th,UINT8_MAX,cv::THRESH_BINARY);
}

void stampit::Image::set_color(uint8_t r, uint8_t g, uint8_t b) {
    if (mat.type() != CV_8UC1) {
        logger->error("set_color() requires a binarized CV_8UC1 image");
        return;
    }
    // 先复制一份灰度图作为 mask
    cv::Mat mask = mat.clone();
    // 再把 mat 转成 BGR
    cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
    cv::Vec3b color(b, g, r);
    for (int y = 0; y < mat.rows; ++y) {
        cv::Vec3b* row = mat.ptr<cv::Vec3b>(y);
        const uchar* m = mask.ptr<uchar>(y); // ✔ 这是正确的 mask

        for (int x = 0; x < mat.cols; ++x) {
            if (m[x] == 0) {
                row[x] = color;
            }
        }
    }
}


void stampit::Image::load(std::string_view path) {
    mat = cv::imread(path.data());
}

void stampit::Image::save(std::string_view path) const {
    cv::imwrite(path.data(),mat);
}

void stampit::Image::save_with_transparent_pix(std::string_view path,int x,int y,int w,int h) const {
    cv::Rect roi(x,y,w,h);
    cv::Mat pic = mat(roi).clone();
    // 确保有 alpha 通道
    if (pic.channels() == 3) {
        cv::cvtColor(pic, pic, cv::COLOR_BGR2BGRA);
    }
    // 将白色像素替换为透明
    for (int y = 0; y < pic.rows; ++y) {
        cv::Vec4b* line = pic.ptr<cv::Vec4b>(y);
        for (int x = 0; x < pic.cols; ++x) {
            cv::Vec4b& px = line[x];
            if (px[0] == 255 && px[1] == 255 && px[2] == 255) {
                px[3] = 0; // alpha 通道设为透明
            }
        }
    }
    cv::imwrite(path.data(),pic);
}

bool stampit::Image::empty() const {
    return mat.empty();
}

int stampit::Image::width() const {
    return mat.cols;
}

int stampit::Image::height() const {
    return mat.rows;
}

uchar* stampit::Image::data() const {
    return mat.data;
}

stampit::Image& stampit::Image::operator=(Image& img) {
    mat = img.mat;
    return *this;
}
