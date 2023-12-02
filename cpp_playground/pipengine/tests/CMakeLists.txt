cmake_minimum_required(VERSION 3.0.0)
project(pipengine VERSION 0.1.0)

add_executable(main_test hello_world.cpp)
target_link_libraries(main_test PUBLIC pipengine)
add_test(NAME main COMMAND test_program)