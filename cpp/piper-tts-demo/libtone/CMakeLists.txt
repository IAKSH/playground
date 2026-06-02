find_package(spdlog REQUIRED)

file(GLOB_RECURSE UI_FILES ui/*.ui)
file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE HPP_FILES include/*.hpp)
file(GLOB_RECURSE H_FILES include/*.h)

add_library(tone ${CPP_FILES} ${HPP_FILES} ${H_FILES})
add_dependencies(tone piper httplib)
target_link_libraries(tone PUBLIC piper asound spdlog::spdlog)
target_include_directories(tone PUBLIC include)
set_target_properties(tone PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)