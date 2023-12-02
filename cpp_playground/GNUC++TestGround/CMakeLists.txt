cmake_minimum_required(VERSION 3.0.0)
project(gnuc++test VERSION 0.1.0)

add_executable(gnuc++test 7-8-4.c)
target_compile_features(gnuc++test PUBLIC cxx_std_20)
target_link_libraries(gnuc++test PUBLIC pthread)
add_test(NAME gnuc++test
    COMMAND build/gnuc++test.exe)