cmake_minimum_required(VERSION 3.5.0)
project(knn VERSION 0.1.0 LANGUAGES C CXX)

find_package(spdlog REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(knn
    main.cpp
    dataset.hpp
    dataset.cpp
    knn.hpp
    knn.cpp
    pca.cpp
    pca.hpp
)

target_include_directories(knn PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(knn ${OpenCV_LIBS} spdlog::spdlog)

set_target_properties(knn PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)