cmake_minimum_required(VERSION 3.0.0)
project(sort_test VERSION 0.1.0)

add_library(sorts INTERFACE)
target_include_directories(sorts INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_sources(sorts
    INTERFACE ${CMAKE_CURRENT_LIST_DIR}/bubble_sort.h
)