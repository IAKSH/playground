cmake_minimum_required(VERSION 3.0.0)
project(c_delete_range_from_ordered_linkedlist VERSION 0.1.0 LANGUAGES C)

include(CTest)
enable_testing()

add_executable(c_std_ceil_floor std_ceil_floor.c)
add_executable(c_linkedlist_remove_range linkedlist_remove_range.c)
add_executable(c_std_strlen std_strlen.c)
add_executable(c_strange_none_huffman strange_none_huffman.c)
add_executable(c_implicit_float_2_int implicit_float_2_int.c)
add_executable(c_check_even_using_logical_and check_even_using_logical_and.c)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
