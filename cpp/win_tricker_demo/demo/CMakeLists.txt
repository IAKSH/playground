cmake_minimum_required(VERSION 3.10.0)
project(win_tricker_demo VERSION 0.1.0 LANGUAGES C CXX)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp)
add_executable(tricker_demo ${sources})
set_target_properties(tricker_demo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# 强制静态链接
set(CMAKE_EXE_LINKER_FLAGS "-static")

target_include_directories(tricker_demo PUBLIC
    include
)

add_dependencies(tricker_demo trick)

target_link_directories(tricker_demo PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}/trick
)

target_link_libraries(tricker_demo PRIVATE
    trick
)