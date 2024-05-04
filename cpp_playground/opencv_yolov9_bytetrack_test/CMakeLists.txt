cmake_minimum_required(VERSION 3.15)

project(opencv_onnx_test CXX)

find_package(spdlog CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE srcs src/*.cpp include/*.h)

add_executable(opencv_onnx_test ${srcs})

target_link_libraries(opencv_onnx_test PRIVATE
	spdlog::spdlog_header_only
	opencv_ml
	opencv_dnn
	opencv_core
	opencv_flann
	opencv_video
	opencv_videoio
	opencv_highgui
	Eigen3::Eigen
)

target_include_directories(opencv_onnx_test PRIVATE
	${CMAKE_CURRENT_LIST_DIR}/include
)
