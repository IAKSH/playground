cmake_minimum_required(VERSION 3.10.0)
project(stampit VERSION 0.0.1 LANGUAGES C CXX)

include(${CMAKE_SOURCE_DIR}/cmake/deploy_helpers.cmake)

set(BUILD_SHARED_LIBS OFF)

set(CROW_USE_BOOST true)
add_subdirectory(vendor/Crow EXCLUDE_FROM_ALL)

add_subdirectory(stampit)
add_subdirectory(stampit-qt5)
add_subdirectory(stampit-qt6)
add_subdirectory(stampit-qt6-legacy)
add_subdirectory(stampit-webui)

set(STAMPIT_ALL_TARGETS
    stampit
    stampit-qt5
    stampit-qt6
    stampit-qt6-legacy
    stampit-webui
)

set(STAMPIT_QT_TARGETS
    stampit-qt5
    stampit-qt6
    stampit-qt6-legacy
)

# CMake在Linux和Windows上有一些行为差异，比如默认的产物目录结构不同
# 可以这样抹平差异（统一用windows上的方式）
foreach(_t IN LISTS STAMPIT_ALL_TARGETS)
    # 获取源目录相对路径
    get_target_property(_src_dir ${_t} SOURCE_DIR)
    file(RELATIVE_PATH _rel "${CMAKE_SOURCE_DIR}" "${_src_dir}")

    set_target_properties(${_t} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_rel}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_rel}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_rel}"
    )
endforeach()


set(STAMPIT_PACK_TARGET "" CACHE STRING "Which target to package (e.g. stampit-qt6, stampit-webui)")

if(STAMPIT_PACK_TARGET)
    list(FIND STAMPIT_ALL_TARGETS "${STAMPIT_PACK_TARGET}" _found_index)
    if(_found_index EQUAL -1)
        message(WARNING "STAMPIT_PACK_TARGET='${STAMPIT_PACK_TARGET}' not found in known targets. Proceeding but install may fail.")
    endif()
    set(PACK_TARGETS ${STAMPIT_PACK_TARGET})
    set(CPACK_PACKAGE_NAME "${STAMPIT_PACK_TARGET}")
else()
    set(PACK_TARGETS ${STAMPIT_ALL_TARGETS})
    set(CPACK_PACKAGE_NAME "stampit-all")
endif()

# 每个 target 的可执行安装到 bin/<target>
foreach(_t IN LISTS PACK_TARGETS)
    install(TARGETS ${_t} RUNTIME DESTINATION bin/${_t})
endforeach()

if(WIN32)
    # 查找 windeployqt 变体
    find_program(WINDEPLOYQT_QT5_EXECUTABLE windeployqt)
    find_program(WINDEPLOYQT_QT6_EXECUTABLE windeployqt-qt6)

    foreach(_t IN LISTS PACK_TARGETS)
        set(_out_dir "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/${_t}")
        list(FIND STAMPIT_QT_TARGETS "${_t}" _is_qt)
        if(NOT _is_qt EQUAL -1)
            if("${_t}" MATCHES "qt6")
                if(WINDEPLOYQT_QT6_EXECUTABLE)
                    add_windows_runtime_deploy(${_t} "${_out_dir}" WINDEPLOYQT "${WINDEPLOYQT_QT6_EXECUTABLE}")
                elseif(WINDEPLOYQT_QT5_EXECUTABLE)
                    add_windows_runtime_deploy(${_t} "${_out_dir}" WINDEPLOYQT "${WINDEPLOYQT_QT5_EXECUTABLE}")
                else()
                    add_windows_runtime_deploy(${_t} "${_out_dir}")
                endif()
            else()
                if(WINDEPLOYQT_QT5_EXECUTABLE)
                    add_windows_runtime_deploy(${_t} "${_out_dir}" WINDEPLOYQT "${WINDEPLOYQT_QT5_EXECUTABLE}")
                elseif(WINDEPLOYQT_QT6_EXECUTABLE)
                    add_windows_runtime_deploy(${_t} "${_out_dir}" WINDEPLOYQT "${WINDEPLOYQT_QT6_EXECUTABLE}")
                else()
                    add_windows_runtime_deploy(${_t} "${_out_dir}")
                endif()
            endif()
        else()
            add_windows_runtime_deploy(${_t} "${_out_dir}")
        endif()
    endforeach()
else()
    # 非 Windows 平台：只需安装静态资源（install_static_for_target 在 deploy_helpers 中实现）
    foreach(_t IN LISTS PACK_TARGETS)
        install_static_for_target(${_t})
    endforeach()
endif()

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "unknown@unknown.com")
set(CPACK_NO_PREINSTALL_TARGET ON)

include(CPack)
