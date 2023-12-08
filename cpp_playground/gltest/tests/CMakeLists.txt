file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp include/*.hpp)
add_executable(temp_test ${sources})
target_include_directories(temp_test PUBLIC include)

find_package(glfw3 CONFIG REQUIRED)
add_dependencies(temp_test glad)

target_link_libraries(temp_test PUBLIC glfw)
target_link_libraries(temp_test PUBLIC glad)