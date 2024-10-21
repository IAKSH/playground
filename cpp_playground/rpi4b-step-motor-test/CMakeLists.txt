cmake_minimum_required(VERSION 3.5.0)
project(step-motor-test VERSION 0.1.0 LANGUAGES C CXX)

add_executable(step-motor-test)
target_sources(step-motor-test PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/step_motor.cpp
)
target_include_directories(step-motor-test PUBLIC ./include)
target_link_libraries(step-motor-test PUBLIC wiringPi)