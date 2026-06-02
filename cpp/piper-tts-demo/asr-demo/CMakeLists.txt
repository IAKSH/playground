find_package(spdlog REQUIRED)
find_package(SDL2 REQUIRED)

file(GLOB_RECURSE UI_FILES ui/*.ui)
file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE HPP_FILES include/*.hpp)
file(GLOB_RECURSE H_FILES include/*.h)

set(WHISPER_EXAMPLES_DIR "${CMAKE_SOURCE_DIR}/vendor/whisper.cpp/examples")

add_executable(asr_demo
    ${CPP_FILES}
    ${HPP_FILES}
    ${H_FILES}
    ${WHISPER_EXAMPLES_DIR}/common-sdl.cpp
)

add_dependencies(asr_demo whisper)
target_link_libraries(asr_demo PRIVATE spdlog::spdlog whisper SDL2::SDL2)
target_include_directories(asr_demo PUBLIC include PRIVATE ${WHISPER_EXAMPLES_DIR})
set_target_properties(asr_demo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)