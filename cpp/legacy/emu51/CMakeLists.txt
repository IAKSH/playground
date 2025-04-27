cmake_minimum_required(VERSION 3.0.0)
project(emu51 VERSION 0.1.0)

add_executable(emu51 main.cpp register.hpp virt_mcu.hpp virt_mcu.cpp)
target_compile_features(emu51 PUBLIC cxx_std_20)