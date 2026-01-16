# frontend/steampit-webui/CMakeLists.txt

# 前端目录
set(FRONTEND_DIR ${CMAKE_SOURCE_DIR}/stampit-webui/frontend/stampit-webui)
set(FRONTEND_BUILD_DIR ${CMAKE_BINARY_DIR}/frontend_build)
set(FRONTEND_BUILD_OUTPUT ${FRONTEND_DIR}/dist) # vite/react 默认输出目录

# npm 命令在 Windows 下是 npm.cmd
if(WIN32)
  set(NPM_CMD "npm.cmd")
else()
  set(NPM_CMD "npm")
endif()

# 1) npm install target (idempotent)
add_custom_target(frontend_install
  COMMAND ${CMAKE_COMMAND} -E echo "Running ${NPM_CMD} install in ${FRONTEND_DIR}..."
  COMMAND ${NPM_CMD} install
  WORKING_DIRECTORY ${FRONTEND_DIR}
  COMMENT "Install frontend dependencies"
)

# 2) frontend build (production)
# 使用 stamp 文件表示构建完成
set(FRONTEND_STAMP ${FRONTEND_BUILD_DIR}/.build_stamp)

# 收集前端源码文件作为依赖
file(GLOB_RECURSE FRONTEND_SRC
  RELATIVE ${FRONTEND_DIR}
  ${FRONTEND_DIR}/src/*.* ${FRONTEND_DIR}/public/*.*)

add_custom_command(
  OUTPUT ${FRONTEND_STAMP}
  DEPENDS frontend_install ${FRONTEND_SRC}
  COMMAND ${CMAKE_COMMAND} -E echo "Building frontend using npm run build..."

  COMMAND ${NPM_CMD} run build
  WORKING_DIRECTORY ${FRONTEND_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${FRONTEND_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E touch ${FRONTEND_STAMP}
  COMMENT "Frontend production build"
)

add_custom_target(frontend_build DEPENDS ${FRONTEND_STAMP})

# 构建完成后复制输出到 backend_static
add_custom_command(TARGET frontend_build POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Copying frontend build to backend_static..."
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/backend_static
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${FRONTEND_BUILD_OUTPUT} ${CMAKE_BINARY_DIR}/stampit-webui/backend/static
  COMMENT "Copy frontend build to backend_static"
)

# 3) frontend dev (npm start) - 长时间运行，不属于默认构建
add_custom_target(frontend_dev
  COMMAND ${CMAKE_COMMAND} -E echo "Starting frontend dev server (npm start)..."
  COMMAND ${NPM_CMD} start
  WORKING_DIRECTORY ${FRONTEND_DIR}
  COMMENT "Run frontend dev server (interactive)"
)

# 父项目可以通过 add_subdirectory(frontend/stampit-webui) 使用 frontend_build
