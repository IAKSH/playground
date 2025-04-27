cmake_minimum_required(VERSION 3.14)

project(SmartHomeControlTerminal)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_SOURCE_DIR}/ui)

find_package(Qt6 COMPONENTS Widgets Network Charts REQUIRED)

set(PROJECT_SOURCES
    include/mainwindow.h
    include/add_device_dialog.h
    include/debug_terminal.h
    include/temperature_chart.h
    include/map.h
    src/main.cpp
    src/mainwindow.cpp
    src/add_device_dialog.cpp
    src/debug_terminal.cpp
    src/temperature_chart.cpp
    src/map.cpp
)

qt6_wrap_ui(UIS_HDRS
    ui/mainwindow.ui
    ui/add_device_dialog.ui
    ui/debug_terminal.ui
)
qt6_add_resources(RESOURCES resources.qrc)

qt_add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
    ${UIS_HDRS}
    ${RESOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets Qt6::Network Qt6::Charts)

# 复制资源文件到构建目录
add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/images ${CMAKE_BINARY_DIR}/images
)