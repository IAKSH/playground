add_library(dagong SHARED)

target_sources(dagong PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/source/demo.cpp
    ${CMAKE_CURRENT_LIST_DIR}/source/jni.cpp
)

target_include_directories(dagong PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    /usr/lib/jvm/java-17-openjdk-amd64/include
	/usr/lib/jvm/java-17-openjdk-amd64/include/linux
)

target_link_libraries(dagong PUBLIC PUBLIC ${OpenCV_LIBRARIES})