cmake_minimum_required(VERSION 3.0.0)
project(ni VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(freeglut_test ${CMAKE_CURRENT_LIST_DIR}/freeglut_test/main.cpp)

target_link_libraries(freeglut_test PUBLIC freeglut Opengl32 glu32)

add_executable(freealut_test ${CMAKE_CURRENT_LIST_DIR}/freealut_test/main.cpp)

add_executable(stb_test ${CMAKE_CURRENT_LIST_DIR}/stb_test/main.cpp)

target_include_directories(stb_test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/stb_test/stb)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
