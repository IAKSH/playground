cmake_minimum_required(VERSION 3.15)

project(lz4_test CXX)

find_package(lz4 REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(lz4_example lz4_example.cpp)
target_link_libraries(lz4_example PRIVATE lz4::lz4)