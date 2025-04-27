project(app CXX)

add_executable(app
    "src/main.cpp"
)

target_include_directories(app PUBLIC
    "include/"
)

add_dependencies(app PUBLIC dll)

target_link_libraries(app PUBLIC dll)