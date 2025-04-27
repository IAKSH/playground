cmake_minimum_required(VERSION 3.0.0)
project(pipengine VERSION 0.1.0)

add_library(pipengine)

target_sources(pipengine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/core/engine/engine.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/engine/engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/core/engine/mixer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/engine/window.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/engine/opengl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/engine/openal.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/engine/layer.hpp
)

target_sources(pipengine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/core/events/keyboard.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/events/mouse.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/events/window.hpp
)

target_sources(pipengine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/core/objects/image.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/objects/texture.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/objects/frame.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/objects/sample.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/objects/sound.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/objects/music.hpp
)

target_sources(pipengine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/core/physics/ball.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/physics/collision.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/physics/point.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/physics/rotate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/physics/square.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/physics/velocity.hpp
)

target_sources(pipengine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/core/pipes/mixpipe.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/pipes/renpipe.hpp
)

target_sources(pipengine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/misc/ctti.hpp
)

target_include_directories(pipengine PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../concepts
)

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