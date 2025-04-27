file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp include/*.hpp)
add_executable(temp_test ${sources})
target_include_directories(temp_test PUBLIC include)

add_dependencies(temp_test nioes)
target_link_libraries(temp_test PUBLIC nioes)