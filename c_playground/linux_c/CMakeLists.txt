cmake_minimum_required(VERSION 3.5.0)
project(lanqiao VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

file(GLOB C_FILES "*.c")

foreach(C_FILE ${C_FILES})
    get_filename_component(TARGET_NAME ${C_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${C_FILE})
endforeach()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)