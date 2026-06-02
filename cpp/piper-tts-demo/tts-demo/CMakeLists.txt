find_package(spdlog REQUIRED)

file(GLOB_RECURSE UI_FILES ui/*.ui)
file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE HPP_FILES include/*.hpp)
file(GLOB_RECURSE H_FILES include/*.h)

add_executable(tts_demo ${CPP_FILES} ${HPP_FILES} ${H_FILES})
add_dependencies(tts_demo tone)
target_link_libraries(tts_demo PRIVATE spdlog::spdlog tone)
target_include_directories(tts_demo PUBLIC include)
set_target_properties(tts_demo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)