cmake_minimum_required(VERSION 3.0.0)
project(pipeng VERSION 0.1.0)

add_library(pipeng pipe.cpp)

find_package(glad CONFIG REQUIRED)
target_link_libraries(pipengine PUBLIC glad::glad)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(pipengine PUBLIC glfw)

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(pipengine PUBLIC spdlog::spdlog spdlog::spdlog_header_only)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(pipengine PUBLIC imgui::imgui)

find_package(Freetype REQUIRED)
target_link_libraries(pipengine PUBLIC Freetype::Freetype)

find_package(OpenAL CONFIG REQUIRED)
target_link_libraries(pipengine PUBLIC OpenAL::OpenAL)