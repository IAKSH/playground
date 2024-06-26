#include <opencv2/opencv.hpp>
#include "YOLODetector.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "Utils/Utils.h"
#include <opencv2/core.hpp>

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::debug);
    CLI::App app{ "yolo onnx opencv dnn pred description" };
    argv = app.ensure_utf8(argv);
    std::string onnxPath, yamlPath, inputPath;
    bool isImage = false, isVideo = false, isCuda = false;
    int imgSize = 640;
    float threshold = 0.5;
    app.add_option("-w,--weights", onnxPath, "onnx model path");
    app.add_option("-c,--config", yamlPath, "train yolo yaml path, to find classes definition");
    app.add_option("-i,--input", inputPath, "pred input path");
    app.add_option("-z,--size", imgSize, "input size(size * size), default:640 * 640");
    app.add_option("-t,--threshold", threshold, "threshold score, default: 0.5");
    app.add_flag("--image, !--no-image", isImage, "Image inference mode");
    app.add_flag("--video, !--no-video", isVideo, "Video inference mode");
    app.add_flag("--gpu, !--no-gpu", isCuda, "Cuda inference mode");
    CLI11_PARSE(app, argc, argv);

    spdlog::debug("cuda count: {}",cv::cuda::getCudaEnabledDeviceCount());

    if (onnxPath.empty()) {
        spdlog::error("onnx model path is empty !!!, onnx model path: {}", onnxPath);
        return 0;
    }
    else {
        if (!FileUtils::fileIsExist(onnxPath)) {
            spdlog::error("onnx model path is not exist !!!, onnx model path: {}, Please check it", onnxPath);
            return 0;
        }
    }
    if (yamlPath.empty()) {
        spdlog::error("yolo yaml path is empty !!! , yolo yaml path: {}", yamlPath);
        return 0;
    }
    else {
        if (!FileUtils::fileIsExist(yamlPath)) {
            spdlog::error("yolo yaml path is not exist !!!, yolo yaml path: {}, Please check it", yamlPath);
            return 0;
        }
    }
    if (inputPath.empty()) {
        spdlog::error("pred input path is empty !!! , pred input path: {}", inputPath);
        return 0;
    }
    else {
        if (!FileUtils::fileIsExist(inputPath)) {
            spdlog::error("pred input path is not exist !!!, pred input path: {}, Please check it", inputPath);
            return 0;
        }
    }
    if (!isImage && !isVideo) {
        spdlog::error("Please select inference mode");
        return 0;
    }
    if (isCuda) {
        int cudaCount = CUDAUtils::getCUDACount();
        if (cudaCount < 1) {
            spdlog::warn("cuda size: {}, default use cpu, please check device exist gpu", cudaCount);
            isCuda = false;
        }
    }
    spdlog::info("onnx model path: {}", onnxPath);
    spdlog::info("yolo yaml path: {}", yamlPath);
    spdlog::info("pred input path: {}", inputPath);
    spdlog::info("pred img size: {}", imgSize);
    spdlog::info("confidence threshold: {}", threshold);
    spdlog::info("Image inference mode: {}", isImage);
    spdlog::info("Video inference mode: {}", isVideo);
    spdlog::info("Cuda inference mode: {}", isCuda);

    //parse ymal, to find classes definition
    std::map<int, std::string> classNames;

    YAML::Node config = YAML::LoadFile(yamlPath);
    if (!config["names"].IsNull() && config["names"].IsMap()) {
        for (size_t i = 0; i < config["names"].size(); ++i) {
            classNames[i] = config["names"][i].as<std::string>();
        }
    }

    spdlog::info("classNames size: {}", classNames.size());
    if (!classNames.empty()) {
        cv::Mat input = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
        cv::Mat img;
        input.copyTo(img);
        std::shared_ptr<YOLODetector> detector = std::make_shared<YOLODetector>();
        int width = imgSize, height = imgSize;
        detector->initConfig(onnxPath, width, height, threshold, isCuda);
        std::vector<DetectResult> results;
        if (isImage) {
            detector->detect(img, results);
            spdlog::debug("results count:\t{}", results.size());
            for (DetectResult& dr : results)
            {
                cv::Rect box = dr.box;

                box.x = int(box.x);
                box.y = int(box.y);
                box.width = int(box.width);
                box.height = int(box.height);
                std::string tips = classNames[dr.classId];
                tips.append(": ");
                tips.append(std::to_string(dr.score));
                cv::putText(input, tips, cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX,
                    .5, cv::Scalar(255, 0, 0));
                cv::rectangle(input, box, cv::Scalar(0, 0, 255), 2, 8);

                spdlog::debug("score:{}\tid:{}\tx:{},y:{},w:{},h:{}", dr.score, classNames[dr.classId], box.x, box.y, box.width, box.height);
            }
            cv::imshow("OpenCV DNN", input);
            cv::waitKey(0);
        }
        else if (isVideo) {
            cv::VideoCapture cap(inputPath);
            if (!cap.isOpened()) {
                spdlog::error("Error opening video stream or file!! path: {}", inputPath);
                return 0;
            }
            cv::Mat mat;
            while (true) {
                cv::Mat frame;
                cap.read(mat); // 读取新的帧
                frame = mat.clone();
                if (frame.empty()) {
                    break;
                }
                detector->detect(frame, results);
                spdlog::debug("results count:\t{}", results.size());
                for (DetectResult& dr : results)
                {
                    cv::Rect box = dr.box;
                    std::string tips = classNames[dr.classId];
                    tips.append(": ");
                    tips.append(std::to_string(dr.score));
                    cv::putText(frame, tips, cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX,
                        .5, cv::Scalar(255, 0, 0));
                    cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2, 8);

                    spdlog::debug(dr.score);
                }
                cv::imshow("OpenCV DNN", frame);
                cv::waitKey(1);
            }
            //todo:
            spdlog::info("Video inference mode todo");
        }
        results.clear();
    }
    else {
        spdlog::error("yaml parse error! yamlPath: {}", yamlPath);
    }
    return 0;
}
