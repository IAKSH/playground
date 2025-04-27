cmake_minimum_required(VERSION 3.0.0)
project(lanqiao VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# 获取当前目录下的所有 .cpp 文件
file(GLOB CPP_FILES "*.cpp")

# 为每个 .cpp 文件创建一个目标
foreach(CPP_FILE ${CPP_FILES})
    # 获取文件名（不包含扩展名）
    get_filename_component(TARGET_NAME ${CPP_FILE} NAME_WE)
    # 添加可执行文件目标
    add_executable(${TARGET_NAME} ${CPP_FILE})
    # 设置目标属性
    set_target_properties(${TARGET_NAME} PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
endforeach()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
