# backend/CMakeLists.txt

# Source files
file(GLOB_RECURSE CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)
file(GLOB_RECURSE HPP_FILES ${CMAKE_CURRENT_LIST_DIR}/include/*.hpp)

# Executable
add_executable(stampit-webui ${CPP_FILES} ${HPP_FILES})

set_target_properties(stampit-webui PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

add_dependencies(stampit-webui stampit Crow)

# Dependencies: stampit library (adjust target name/path as needed)
# If stampit is built as a target in your superproject, link it; otherwise link system lib
find_library(STAMPIT_LIB NAMES stampit PATHS ${CMAKE_SOURCE_DIR}/../lib NO_DEFAULT_PATH)
if(TARGET stampit)
  target_link_libraries(stampit-webui PRIVATE stampit)
elseif(STAMPIT_LIB)
  target_link_libraries(stampit-webui PRIVATE ${STAMPIT_LIB})
else()
  message(STATUS "Warning: stampit library not found by find_library; ensure it's linked manually")
endif()

# Include dirs
target_include_directories(stampit-webui PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}/../../vendor/Crow/include
)

# Define STATIC_DIR macro to point to build/backend_static by default
# Use generator expression to expand at configure time
target_compile_definitions(stampit-webui PRIVATE STATIC_DIR="static")

# Windows socket libs
if (WIN32)
  target_link_libraries(stampit-webui PRIVATE ws2_32 mswsock)
endif()

# Ensure backend depends on frontend_build so packaging copies static files first if desired
if(FRONTEND_BUILD_TYPE STREQUAL "prod")
  add_dependencies(stampit-webui frontend_build)
else()
  add_dependencies(stampit-webui frontend_dev)
endif()

