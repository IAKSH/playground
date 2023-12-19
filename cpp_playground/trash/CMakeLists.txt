cmake_minimum_required(VERSION 3.0.0)
project(trash VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(cpp_others_huffman_tree others_huffman_tree.cpp)
add_executable(cpp_print_u018 print_u018.cpp)
add_executable(cpp_construction_order construction_order.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
