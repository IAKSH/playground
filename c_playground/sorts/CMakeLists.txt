cmake_minimum_required(VERSION 3.0.0)
project(thread_test VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(c_teacher_hash hash_collision_fix.c)
add_executable(c_teacher_linked_hash hash_collision_fix_linked.c)
add_executable(c_insert_sort insert_sort.c)
add_executable(c_binary_insert_sort binary_insert_sort.c)
add_executable(c_bubble_sort bubble_sort.c)
add_executable(c_shell_sort shell_sort.c)
add_executable(c_select_sort select_sort.c)
add_executable(c_search_from_bst search_from_bst.c)
add_executable(c_temp temp.c)

target_link_libraries(c_temp PUBLIC m)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
