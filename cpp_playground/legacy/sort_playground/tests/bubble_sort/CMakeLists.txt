cmake_minimum_required(VERSION 3.0.0)
project(sort_test VERSION 0.1.0)

add_executable(bubble_sort_f)
add_dependencies(bubble_sort_f sorts)
target_link_libraries(bubble_sort_f sorts)
target_sources(bubble_sort_f PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_f.c
)
add_test(NAME bubble_sort_f COMMAND bubble_sort_f)

add_executable(bubble_sort_d)
add_dependencies(bubble_sort_d sorts)
target_link_libraries(bubble_sort_d sorts)
target_sources(bubble_sort_d PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_d.c
)
add_test(NAME bubble_sort_d COMMAND bubble_sort_d)

add_executable(bubble_sort_i32)
add_dependencies(bubble_sort_i32 sorts)
target_link_libraries(bubble_sort_i32 sorts)
target_sources(bubble_sort_i32 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_i32.c
)
add_test(NAME bubble_sort_i32 COMMAND bubble_sort_i32)

add_executable(bubble_sort_i16)
add_dependencies(bubble_sort_i16 sorts)
target_link_libraries(bubble_sort_i16 sorts)
target_sources(bubble_sort_i16 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_i16.c
)
add_test(NAME bubble_sort_i16 COMMAND bubble_sort_i16)

add_executable(bubble_sort_i8)
add_dependencies(bubble_sort_i8 sorts)
target_link_libraries(bubble_sort_i8 sorts)
target_sources(bubble_sort_i8 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_i8.c
)
add_test(NAME bubble_sort_i8 COMMAND bubble_sort_i8)

add_executable(bubble_sort_u32)
add_dependencies(bubble_sort_u32 sorts)
target_link_libraries(bubble_sort_u32 sorts)
target_sources(bubble_sort_u32 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_u32.c
)
add_test(NAME bubble_sort_u32 COMMAND bubble_sort_u32)

add_executable(bubble_sort_u16)
add_dependencies(bubble_sort_u16 sorts)
target_link_libraries(bubble_sort_u16 sorts)
target_sources(bubble_sort_u16 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_u16.c
)
add_test(NAME bubble_sort_u16 COMMAND bubble_sort_u16)

add_executable(bubble_sort_u8)
add_dependencies(bubble_sort_u8 sorts)
target_link_libraries(bubble_sort_u8 sorts)
target_sources(bubble_sort_u8 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_u8.c
)
add_test(NAME bubble_sort_u8 COMMAND bubble_sort_u8)

add_executable(bubble_sort_universual_i32)
add_dependencies(bubble_sort_universual_i32 sorts)
target_link_libraries(bubble_sort_universual_i32 sorts)
target_sources(bubble_sort_universual_i32 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_universual_i32.c
)
add_test(NAME bubble_sort_universual_i32 COMMAND bubble_sort_universual_i32)

add_executable(bubble_sort_universual_u8)
add_dependencies(bubble_sort_universual_u8 sorts)
target_link_libraries(bubble_sort_universual_u8 sorts)
target_sources(bubble_sort_universual_u8 PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_universual_u8.c
)
add_test(NAME bubble_sort_universual_u8 COMMAND bubble_sort_universual_u8)

add_executable(bubble_sort_universual_strcut)
add_dependencies(bubble_sort_universual_strcut sorts)
target_link_libraries(bubble_sort_universual_strcut sorts)
target_sources(bubble_sort_universual_strcut PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/bubble_sort_universual_strcut.c
)
add_test(NAME bubble_sort_universual_strcut COMMAND bubble_sort_universual_strcut)