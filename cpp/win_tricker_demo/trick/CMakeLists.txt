cmake_minimum_required(VERSION 3.10.0)
project(win_tricker_demo VERSION 0.1.0 LANGUAGES C CXX)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp source/*.c)
add_library(trick ${sources})
set_target_properties(trick PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(trick PUBLIC
    include
)