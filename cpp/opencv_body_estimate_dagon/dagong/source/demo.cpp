#include <iostream>
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define PRINT_AVERAGE_PIXEL_LENGTH

static cv::Mat graylize(const cv::Mat& img) {
    cv::Mat gray = img;
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            cv::Vec3b& pixel = gray.at<cv::Vec3b>(i, j);
            uint8_t blue = pixel[0];
            uint8_t green = pixel[1];
            uint8_t red = pixel[2];

            float gray = 0.299 * red + 0.587 * green + 0.114 * blue;
            pixel[0] = pixel[1] = pixel[2] = cv::saturate_cast<uchar>(gray);
        }
    }
    return gray;
}

static cv::Mat enhance(const cv::Mat& img) {
    cv::Mat enhanced = img;
    // 原论文似乎有这俩完全是magic number的意思，可能需要通过某种算法（或者手动）对每一张图片特化
    // 手动调了几张图，对最终结果的影响确实很大
    constexpr float MULT = 1.64516f,ADD = -165.0f;
    for(int i = 0;i < enhanced.rows;i++) {
        for(int j = 0;j < enhanced.cols;j++) {
            cv::Vec3b& pixel = enhanced.at<cv::Vec3b>(i, j);
            pixel[0] = pixel[1] = pixel[2] = cv::saturate_cast<uchar>(pixel[0] * MULT + ADD);
        }
    }
    return enhanced;
}

static cv::Mat gaussian_filter(const cv::Mat& img) {
    cv::Mat blur_img;
    cv::GaussianBlur(img, blur_img, cv::Size(9, 9), 1000000000.0);
    return blur_img;
}

// 计算图像的灰度直方图
static std::vector<int> calcHist(const cv::Mat& img) {
    std::vector<int> histogram(256, 0);
    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            histogram[(int)img.at<uchar>(i,j)]++;
        }
    }
    return histogram;
}

static cv::Mat apply_threshold(const cv::Mat& img, double thresh) {
    cv::Mat result;
    cv::threshold(img, result, thresh, 255, cv::THRESH_BINARY);
    return result;
}

static cv::Mat improved_otsu(const cv::Mat _img) {
    // 将图像转换为灰度图像
    cv::Mat img = _img;
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    int total = img.rows * img.cols;
    std::vector<int> histogram = calcHist(img);

    /*
    double sum = 0;
    for(int i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }

    double sumB = 0;
    int wB = 0;
    int wF = 0;

    double varMax = 0;
    double threshold = 0;

    for(int i = 0 ; i < 256 ; i++) {
        wB += histogram[i];
        if(wB == 0)
            continue;

        wF = total - wB;
        if(wF == 0)
            break;

        sumB += i * histogram[i];

        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        float mT = (float)sum / total;

        double varBetween = wB * wF * (mB - mF) * (mB - mF);
        //double varBetween = wB * (mB - mT) * (mB - mT) + wF * log10(2 - wF) * (mF - mT) * (mF - mT);

        if(varBetween > varMax) {
            varMax = varBetween;
            threshold = i;
        }
    }
    */

    cv::Mat otsu_img;
    double threshold = cv::threshold(
        img, otsu_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU
    );

    // 计算权重
    double weight = 0.0;
    for (int i = 0; i < threshold; i++) {
        weight += histogram[i];
    }

    // 计算偏移量
    double offset = weight / total;

    // 计算最佳阈值
    double best_thresh_val = threshold + offset;

    std::cout << best_thresh_val << '\t' << threshold << '\t' << offset << '\t' << weight << '\n';

    return apply_threshold(img,best_thresh_val / 2);
}

static cv::Mat otsu(const cv::Mat& img) {
    cv::Mat out_img;
    cv::cvtColor(img, out_img, cv::COLOR_BGR2GRAY);
    double otsu_thresh_val = cv::threshold(
        img, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU
    );
    std::cout << "otsu_thresh_val = " << otsu_thresh_val << '\n';
    return out_img;
}

static cv::Mat edge(const cv::Mat& img) {
    cv::Mat edge;
    cv::Canny(img, edge, 50, 150);
    return edge;
}

static json output;

static cv::Mat mark_part(const cv::Mat& img,std::string part_name,const cv::Rect& max_rect,float p,float l,float h) {
    /*
         女                     男
    裆部 0.444（0.407-0.481）   0.441（0.401-0.481）
    颈部 0.813（0.792-0.833）   0.815（0.793-0.835）
    肩部 0.807（0.783-0.831）   0.808（0.781-0.836）
    胸部 0.712（0.674-0.751）   0.716（0.688-0.745）
    腰部 0.615（0.578-0.652）   0.612（0.575-0.649）
    臀部 0.551（0.493-0.609）   0.567（0.510-0.624）
    */
    cv::Mat out = img;
    cv::Rect rect = max_rect;

    rect.height = max_rect.height * (h - l);
    rect.y = max_rect.y + max_rect.height * (1 - p);
    cv::rectangle(out, rect, cv::Scalar(0, 255, 0), 2);

#ifdef PRINT_AVERAGE_PIXEL_LENGTH
    // 在rect的中心开始逐行向左右搜索边界
    cv::Mat rect_mat = img(rect);
    int total_pixel_length = 0;
    int count = 0;
    for(int i = 0; i < rect_mat.rows; i++) {
        bool state = false;
        std::vector<int> line_pixel_length;
        for(int j = 0; j < rect_mat.cols; j++) {
            bool flag = true;
            while(rect_mat.at<cv::Vec3b>(i, ++j)[0] != 0) {
                if(j >= rect_mat.cols) {
                    flag = false;
                    break;
                }
                if(state) {
                    state = false;
                }
                ++j;
            }
            if(!flag) {
                break;
            }
            if(!state) {
                line_pixel_length.emplace_back(0);
                state = true;
            }
            ++line_pixel_length.back();
        }

        total_pixel_length += *std::max_element(line_pixel_length.begin(),line_pixel_length.end());
        ++count;
    }

    // 计算并打印所有行加起来的平均像素长度
    static int json_count = 0;
    if (count > 0) {
        //float average_pixel_length = static_cast<float>(total_pixel_length) / count;
        //std::cout << "average pixel length: " << average_pixel_length << '\t' << "converted (160cm): " << 160.0 / max_rect.width * average_pixel_length << "cm" << '\n';
        float average_pixel_length = static_cast<float>(total_pixel_length) / count;
        json node;
        node["part"] = part_name;
        node["pixel"] = average_pixel_length;
        node["converted"] = 160.0 / max_rect.width * average_pixel_length;
        output[json_count++] = node;
    }

    return out;
#endif

    return out;
}

static cv::Mat mark_point(const cv::Mat& img,const cv::Rect& max_rect,float p,float l,float h) {
    cv::Mat with_point;
    img.copyTo(with_point);
    cv::Rect rect = max_rect;
    rect.height = max_rect.height * (h - l);
    rect.y = max_rect.y + max_rect.height * (1 - p);
    with_point = with_point(rect);

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    // 角点检测参数
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    // 使用Harris角点检测
    cv::cornerHarris(with_point, dst, blockSize, apertureSize, k);

    // 归一化以便于结果的可视化
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // 将检测到的角点标记在图像上
    for(int i = 0; i < dst_norm.rows ; i++) {
        for(int j = 0; j < dst_norm.cols; j++) {
            if((int) dst_norm.at<float>(i,j) > 200) {
                cv::circle(dst_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(0), 2, 8, 0);
            }
        }
    }

    return dst_norm_scaled;
}

std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_decode(const std::string &in) {
    std::string out;

    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[base64_chars[i]] = i;

    int val = 0, valb = -8;
    for (uchar c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::string body_estimate_demo(const std::string& base64_img) {

    std::cout << "C++ received base64:\n" << base64_img << '\n';

    // Decode the base64 image
    std::string decoded_img = base64_decode(base64_img);
    std::vector<uchar> img_data(decoded_img.begin(), decoded_img.end());

    // Convert the data to cv::Mat
    cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "can't read" << std::endl;
        exit(1);
    }

    auto gray = graylize(img);
    cv::imwrite("./out/gray.jpg", gray);
    auto enhanced1 = enhance(gray);
    auto guassian = gaussian_filter(enhanced1);
    //auto enhanced2 = enhance(enhanced1);
    //cv::imwrite("/home/lain/Desktop/enhanced2.jpg", enhanced2);
    //auto guassian = gaussian_filter(enhanced2);
    cv::imwrite("./out/guassian.jpg", guassian);
    auto otsu = improved_otsu(guassian);
    cv::imwrite("./out/otsu.jpg", otsu);
    auto edged = edge(otsu);
    cv::imwrite("./out/edged.jpg", edged);

    // 找到轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edged, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 画出最大的外接矩形
    cv::Rect max_rect;
    for(int i = 0;i < contours.size();i++) {
        const auto&& rect = cv::boundingRect(contours[i]);
        if(rect.area() > max_rect.area()) {
            max_rect = rect;
        }
    }

    cv::Mat edged_rgb,with_rect;
    cv::cvtColor(edged, edged_rgb, cv::COLOR_GRAY2BGR);
    edged_rgb.copyTo(with_rect);
    cv::rectangle(with_rect, max_rect, cv::Scalar(0, 255, 0), 2);
    cv::imwrite("./out/with_rect.jpg", with_rect);

    cv::Mat parts;
    edged_rgb.copyTo(parts);

    auto crotch = mark_part(parts,"crotch",max_rect,0.441,0.401,0.481);
    cv::imwrite("./out/crotch.jpg", crotch);
    edged_rgb.copyTo(parts);

    //auto neck = mark_part(parts,max_rect,0.815,0.793,0.835);
    //cv::imwrite("/home/lain/Desktop/neck.jpg", neck);
    //edged_rgb.copyTo(parts);

    auto shoulder = mark_part(parts,"shoulder",max_rect,0.808,0.781,0.836);
    cv::imwrite("./out/shoulder.jpg", shoulder);
    edged_rgb.copyTo(parts);

    auto chest = mark_part(parts,"chest",max_rect,0.716,0.688,0.745);
    cv::imwrite("./out/chest.jpg", chest);
    edged_rgb.copyTo(parts);

    auto waist = mark_part(parts,"waist",max_rect,0.612,0.575,0.649);
    cv::imwrite("./out/waist.jpg", waist);
    edged_rgb.copyTo(parts);

    auto buttocks = mark_part(parts,"buttocks",max_rect,0.567,0.510,0.624);
    cv::imwrite("./out/buttocks.jpg", buttocks);
    edged_rgb.copyTo(parts);

    //auto waist_with_points = mark_point(edged,max_rect,0.567,0.510,0.624);
    //cv::imwrite("/home/lain/Desktop/waist_with_points.jpg", waist_with_points);

    return output.dump();
}
