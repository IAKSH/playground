file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp include/*.hpp)
add_executable(cppoes_test ${sources})
target_include_directories(cppoes_test PRIVATE include)

add_dependencies(cppoes_test cppoes)
#find_package(imgui CONFIG REQUIRED)
target_link_libraries(cppoes_test PRIVATE cppoes)
#target_link_libraries(cppoes_test PRIVATE imgui::imgui)