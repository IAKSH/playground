cmake_minimum_required(VERSION 3.0.0)
project(ModuleTest VERSION 0.1.0)

add_executable(ModuleTest main.cpp mylib.cpp mylib.ixx)
target_compile_features(ModuleTest PUBLIC cxx_std_20)
