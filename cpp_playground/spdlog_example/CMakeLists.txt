cmake_minimum_required(VERSION 3.15)

project(spdlog_test CXX)

find_package(spdlog REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(default_logger default_logger.cpp)
target_link_libraries(default_logger PRIVATE spdlog::spdlog)

add_executable(custom_logger custom_logger.cpp)
target_link_libraries(custom_logger PRIVATE spdlog::spdlog)

add_executable(stdout_sink stdout_sink.cpp)
target_link_libraries(stdout_sink PRIVATE spdlog::spdlog)

add_executable(file_sink file_sink.cpp)
target_link_libraries(file_sink PRIVATE spdlog::spdlog)