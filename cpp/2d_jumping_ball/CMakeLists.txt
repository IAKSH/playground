cmake_minimum_required(VERSION 3.15)

project(2d_jumping_ball CXX)

find_package(OpenAL CONFIG REQUIRED)
find_package(mpg123 CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(2d_jumping_ball
    "src/main.cpp"
    "src/graphics.cpp"
    "src/audio.cpp"
    "src/physics.cpp"
)

target_include_directories(2d_jumping_ball PUBLIC
    "include/"
)

target_link_libraries(2d_jumping_ball PRIVATE
    spdlog::spdlog
    OpenAL::OpenAL
    MPG123::libmpg123
    MPG123::libout123
    MPG123::libsyn123
    imgui::imgui
    glfw
    glad::glad
    glm::glm
)

add_custom_command(TARGET 2d_jumping_ball POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/resource/hit.mp3
                       $<TARGET_FILE_DIR:2d_jumping_ball>/hit.mp3)