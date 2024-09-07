project(inject CXX)

add_executable(inject
    "src/main.cpp"
)

target_include_directories(inject PUBLIC
    "include/"
)