cmake_minimum_required(VERSION 3.0.0)
project(cppadv VERSION 0.1.0)

add_subdirectory(GLREN)

add_executable(cppadv main.cpp)
target_compile_features(cppadv PUBLIC cxx_std_20)