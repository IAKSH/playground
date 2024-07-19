add_executable(dagong)

target_sources(dagong PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/source/main.cpp
)

target_include_directories(dagong PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include ${OpenCV_INCLUDE_DIRS})

target_link_libraries(dagong PUBLIC PUBLIC ${OpenCV_LIBRARIES})