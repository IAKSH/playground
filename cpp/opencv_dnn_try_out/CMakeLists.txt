cmake_minimum_required(VERSION 3.25)
project(opencv-dnn-try-out)

set(CMAKE_CXX_STANDARD 17)

find_package(CLI11 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(opencv-dnn-try-out main.cpp
        YOLODetector.cpp
        YOLODetector.h
        Utils/Utils.cpp
        Utils/Utils.h
)
target_link_libraries(opencv-dnn-try-out
        CLI11::CLI11
        spdlog::spdlog_header_only
        ${OpenCV_LIBS}
        yaml-cpp
)