find_package(Boost 1.64 COMPONENTS system date_time REQUIRED)

add_executable(viewer)
add_dependencies(viewer glad)
target_sources(viewer PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/source/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/source/mqtt.cpp
    )
target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../vendor/mqtt_cpp/include)
target_link_libraries(viewer PUBLIC glad glfw Boost::boost Boost::system Boost::date_time)