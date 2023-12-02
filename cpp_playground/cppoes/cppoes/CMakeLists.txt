file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp include/*.hpp)
add_library(cppoes ${sources})
target_include_directories(cppoes PRIVATE include)

add_subdirectory(vendor/glad)