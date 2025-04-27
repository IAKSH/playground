project(dll CXX)

add_library(dll SHARED
    "src/mydll.cpp"
    "include/mydll.h"
)

target_include_directories(dll PUBLIC
    "include/"
)

target_compile_definitions(dll PRIVATE MYDLL_EXPORTS)