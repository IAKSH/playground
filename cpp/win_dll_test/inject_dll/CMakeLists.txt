project(inject_dll CXX)

add_library(inject_dll SHARED
    "src/inject_dll.cpp"
    "include/inject_dll.h"
)

target_include_directories(inject_dll PUBLIC
    "include/"
)

target_compile_definitions(inject_dll PRIVATE MYDLL_EXPORTS)