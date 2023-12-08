file(GLOB_RECURSE sources CONFIGURE_DEPENDS source/*.cpp include/*.hpp)
add_library(gl ${sources})
target_include_directories(gl PUBLIC include)

add_dependencies(gl glad)

target_link_libraries(gl PUBLIC glad)

find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
	
target_link_libraries(gl PUBLIC glfw)
target_link_libraries(gl PUBLIC imgui::imgui)
target_link_libraries(gl PUBLIC glm::glm)
target_link_libraries(gl PUBLIC assimp::assimp)
target_link_libraries(gl PUBLIC spdlog::spdlog)
target_include_directories(gl PRIVATE ${Stb_INCLUDE_DIR})