cmake_minimum_required(VERSION 3.0.0)
project(hamming_code VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(7-4bit_decode 7-4bit_decode.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
