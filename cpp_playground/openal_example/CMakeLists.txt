cmake_minimum_required(VERSION 3.15)

project(openal_test CXX)

find_package(OpenAL CONFIG REQUIRED)
find_package(mpg123 CONFIG REQUIRED)
find_package(spdlog REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(main_test main.cpp)
target_link_libraries(main_test PRIVATE
    spdlog::spdlog
    OpenAL::OpenAL
    MPG123::libmpg123
    MPG123::libout123
    MPG123::libsyn123
)

# 添加自定义命令
add_custom_command(TARGET main_test POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/a.mp3
                       $<TARGET_FILE_DIR:main_test>/a.mp3)
