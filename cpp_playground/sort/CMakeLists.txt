cmake_minimum_required(VERSION 3.0.0)
project(sort VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)

add_executable(cpp_std_sort std_sort.cpp)
add_executable(cpp_std_stable_sort std_stable_sort.cpp)
add_executable(cpp_std_partial_sort std_partial_sort.cpp)
add_executable(cpp_std_partial_sort_copy std_partial_sort_copy.cpp)
add_executable(cpp_std_ranges_sort std_ranges_sort.cpp)
add_executable(cpp_std_ranges_stable_sort std_ranges_stable_sort.cpp)
add_executable(cpp_std_ranges_partial_sort std_ranges_partial_sort.cpp)
add_executable(cpp_std_ranges_partial_sort_copy std_ranges_partial_sort_copy.cpp)
add_executable(cpp_heap_sort_using_stl heap_sort_using_stl.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
