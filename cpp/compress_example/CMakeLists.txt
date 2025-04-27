cmake_minimum_required(VERSION 3.15)

project(compress_test CXX)

find_package(lz4 REQUIRED)
find_package(liblzma CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(lz4_example lz4_example.cpp)
target_link_libraries(lz4_example PRIVATE lz4::lz4)

add_executable(lzma_example lzma_example.cpp)
target_link_libraries(lzma_example PRIVATE liblzma::liblzma)

add_executable(zstd_example zstd_example.cpp)
target_link_libraries(zstd_example PRIVATE $<IF:$<TARGET_EXISTS:zstd::libzstd_shared>,zstd::libzstd_shared,zstd::libzstd_static>)