find_package(spdlog REQUIRED)

file(GLOB_RECURSE UI_FILES ui/*.ui)
file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE HPP_FILES include/*.hpp)
file(GLOB_RECURSE H_FILES include/*.h)

add_executable(llm_tts_demo ${CPP_FILES} ${HPP_FILES} ${H_FILES})
add_dependencies(llm_tts_demo piper httplib nlohmann_json tone)
target_link_libraries(llm_tts_demo PRIVATE spdlog::spdlog httplib::httplib nlohmann_json::nlohmann_json tone)
target_include_directories(llm_tts_demo PUBLIC include)
set_target_properties(llm_tts_demo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)