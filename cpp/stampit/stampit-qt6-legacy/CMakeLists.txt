find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)
find_package(OpenCV REQUIRED)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)


file(GLOB_RECURSE UI_FILES ui/*.ui)
file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE HPP_FILES include/*.hpp)
add_executable(stampit-qt6-legacy ${CPP_FILES} ${HPP_FILES} ${UI_FILES})
set_property(TARGET stampit-qt6-legacy PROPERTY AUTOUIC_SEARCH_PATHS ${CMAKE_SOURCE_DIR}/stampit-qt6-legacy/ui)

if(MSVC OR MINGW)
set_target_properties(stampit-qt6-legacy PROPERTIES
    WIN32_EXECUTABLE $<$<CONFIG:Release>:TRUE>
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
else()
set_target_properties(stampit-qt6-legacy PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
endif()

target_include_directories(stampit-qt6-legacy PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(stampit-qt6-legacy PRIVATE
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
    ${OpenCV_LIBS}
)

#if(MINGW OR MSVC)
#    # 查找 windeployqt
#    find_program(WINDEPLOYQT_EXECUTABLE windeployqt
#        HINTS "${Qt6_DIR}/../../../bin" "${Qt6_DIR}/bin")
#    if(NOT WINDEPLOYQT_EXECUTABLE)
#        message(FATAL_ERROR "windeployqt not found")
#    endif()
#    # 安装主程序
#    install(TARGETS stampit-qt6-legacy RUNTIME DESTINATION bin)
#    # 在安装后运行 windeployqt
#    install(CODE "
#        message(STATUS \"Running windeployqt for Qt6 dependencies...\")
#        execute_process(
#            COMMAND \"${WINDEPLOYQT_EXECUTABLE}\"
#                    --no-compiler-runtime
#                    --no-opengl-sw
#                    --verbose 1
#                    \"$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/stampit-qt6-legacy.exe\"
#            WORKING_DIRECTORY \"$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin\"
#            RESULT_VARIABLE result
#        )
#        if(NOT result EQUAL 0)
#            message(FATAL_ERROR \"windeployqt failed with exit code \${result}\")
#        endif()
#        message(STATUS \"windeployqt completed successfully\")
#    ")
#    
#    if(MINGW)
#        # 查找 Python 和 objdump (MinGW)
#        find_program(PYTHON_EXECUTABLE python REQUIRED)
#        find_program(OBJDUMP_EXECUTABLE objdump REQUIRED)
#        # 推断 MinGW 根目录
#        get_filename_component(MINGW_BIN_DIR "${OBJDUMP_EXECUTABLE}" DIRECTORY)
#        get_filename_component(MINGW_ROOT "${MINGW_BIN_DIR}" DIRECTORY)
#        
#        # 在安装后运行 MinGW DLL 收集脚本
#        install(CODE "
#            message(STATUS \"Collecting MinGW runtime dependencies...\")
#            if(NOT EXISTS \"${CMAKE_SOURCE_DIR}/scripts/mingw_collect_deps.py\")
#                message(WARNING \"MinGW dependency collection script not found at: ${CMAKE_SOURCE_DIR}/scripts/mingw_collect_deps.py\")
#            else()
#                execute_process(
#                    COMMAND \"${PYTHON_EXECUTABLE}\"
#                            \"${CMAKE_SOURCE_DIR}/scripts/mingw_collect_deps.py\"
#                            --mingw \"${MINGW_ROOT}\"
#                            --input \"$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/stampit-qt6-legacy.exe\"
#                            --output \"$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin\"
#                    RESULT_VARIABLE result
#                )
#                if(NOT result EQUAL 0)
#                    message(WARNING \"MinGW dependency collection failed with exit code \${result}\")
#                else()
#                    message(STATUS \"MinGW dependencies collected successfully\")
#                endif()
#            endif()
#        ")
#    elseif(MSVC)
#        # 对于 MSVC + vcpkg，安装构建目录中的所有 DLL
#        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
#            DESTINATION bin
#            FILES_MATCHING 
#            PATTERN "*.dll"
#            PATTERN "*.pdb" EXCLUDE
#            PATTERN "vcpkg_installed" EXCLUDE
#            PATTERN "Testing" EXCLUDE
#            PATTERN "stampit-qt6-legacy_autogen" EXCLUDE
#            PATTERN "CMakeFiles" EXCLUDE
#            PATTERN ".qt" EXCLUDE
#            PATTERN ".cmake" EXCLUDE
#        )
#        message(STATUS "MSVC/vcpkg DLLs will be installed from: ${CMAKE_CURRENT_BINARY_DIR}/\$<CONFIG>/")
#    endif()
#
#    set(CPACK_PACKAGE_NAME "stampit-qt6-legacy")
#    set(CPACK_PACKAGE_VENDOR "unknown")
#    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "unknown")
#    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
#    set(CPACK_PACKAGE_CONTACT "unknown@unknown.com")
#    
#    set(CPACK_GENERATOR "ZIP")
#    # ...existing code...
#endif()