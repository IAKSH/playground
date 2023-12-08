file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.c include/*.h)
add_library(glad ${sources})
target_include_directories(glad PUBLIC include)