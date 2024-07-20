add_library(glad)
target_sources(glad PUBLIC ${CMAKE_CURRENT_LIST_DIR}/source/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)