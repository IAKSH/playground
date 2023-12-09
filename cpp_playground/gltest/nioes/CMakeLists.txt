file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp include/*.hpp)
add_library(nioes ${sources})
target_include_directories(nioes PUBLIC include)

add_dependencies(nioes glad)
add_dependencies(nioes tinygltf)

target_link_libraries(nioes PUBLIC glad)
target_link_libraries(nioes INTERFACE tinygltf)

find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
	
target_link_libraries(nioes PUBLIC glfw)
target_link_libraries(nioes PUBLIC imgui::imgui)
target_link_libraries(nioes PUBLIC glm::glm)
target_link_libraries(nioes PUBLIC tinygltf)
target_link_libraries(nioes PUBLIC spdlog::spdlog)
target_include_directories(nioes PRIVATE ${Stb_INCLUDE_DIR})