file(GLOB_RECURSE sources CONFIGURE_DEPENDS include/*.h include/*.hpp)
add_library(tinygltf INTERFACE ${sources})
target_include_directories(tinygltf INTERFACE include)