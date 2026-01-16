# stampit-webui/CMakeLists.txt
# 允许用户通过 -DFRONTEND_BUILD_TYPE=dev 来控制前端是 dev 还是 prod
if(NOT DEFINED FRONTEND_BUILD_TYPE)
  set(FRONTEND_BUILD_TYPE "prod" CACHE STRING "frontend build type: prod or dev")
endif()

# 子项目：frontend 与 backend
add_subdirectory(frontend/stampit-webui)
add_subdirectory(backend)

# 运行目标：确保 frontend_build 与 后端可执行 已构建，然后运行后端
#add_custom_target(run
#  DEPENDS stampit-webui frontend_build
#  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target stampit-webui
#  COMMAND ${CMAKE_BINARY_DIR}/stampit-webui
#  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#  COMMENT "Build and run stampit-webui (backend) with frontend static files"
#)

# 安装 / 打包：把后端可执行与静态文件安装到 package
#install(TARGETS stampit-webui DESTINATION bin)
#install(DIRECTORY ${CMAKE_BINARY_DIR}/backend_static DESTINATION share/${PROJECT_NAME}/static)

#add_windows_runtime_deploy(stampit-webui)

# ...existing code...
