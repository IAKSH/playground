cmake_minimum_required(VERSION 3.10.0)
project(crtp VERSION 0.1.0 LANGUAGES C CXX)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp)
add_executable(crtp ${sources})
set_target_properties(crtp PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(crtp PUBLIC
    include
)
