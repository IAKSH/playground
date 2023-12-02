cmake_minimum_required(VERSION 3.0.0)
project(sort_test VERSION 0.1.0)

add_executable(counting_sort_heap_i32)
add_dependencies(counting_sort_heap_i32 sorts)
target_link_libraries(counting_sort_heap_i32 sorts)
target_sources(counting_sort_heap_i32 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/counting_sort_heap_i32.c
)
add_test(NAME counting_sort_heap_i32 COMMAND counting_sort_heap_i32)