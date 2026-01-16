find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE HPP_FILES include/*.hpp)
add_library(stampit ${CPP_FILES} ${HPP_FILES})

set_target_properties(stampit PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(stampit PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(stampit PRIVATE
    fmt::fmt
    ${OpenCV_LIBS}
)
