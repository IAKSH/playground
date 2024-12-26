cmake_minimum_required(VERSION 3.5.0)
project(kmeans VERSION 0.1.0 LANGUAGES C CXX)

find_package(spdlog REQUIRED)

add_executable(kmeans
    main.cpp
)

target_include_directories(kmeans PUBLIC spdlog::spdlog)
target_link_libraries(kmeans spdlog::spdlog)

set_target_properties(kmeans PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)