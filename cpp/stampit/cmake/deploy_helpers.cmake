# cmake/deploy_helpers.cmake

# Helper: find_static_dir_for_target(<target> <out_var>)
# 查找 static 目录：优先基于构建产物（exe）所在目录的同级 static。
# 在构建目录中递归查找与 target 同名的可执行（Windows 同时查找 .exe），
# 对每个找到的 exe 检查其所在目录是否包含 static，若找到则返回该路径。
# 如果没有找到任何 exe 同级的 static，则返回空（不回退到全局候选路径）。
function(find_static_dir_for_target target out_var)
    set(_t "${target}")

    # 在构建目录中递归查找可执行候选
    set(_exe_candidates "")
    if(WIN32)
        file(GLOB_RECURSE _exe_candidates
            "${CMAKE_BINARY_DIR}/**/${_t}.exe"
        )
    else()
        # 在类 Unix 上可执行通常没有扩展名，查找同名文件
        file(GLOB_RECURSE _exe_candidates
            "${CMAKE_BINARY_DIR}/**/${_t}"
        )
    endif()

    set(_found "")
    foreach(_exe_path IN LISTS _exe_candidates)
        # 规范化为绝对路径
        get_filename_component(_abs_exe "${_exe_path}" ABSOLUTE)
        get_filename_component(_exe_dir "${_abs_exe}" DIRECTORY)
        if(EXISTS "${_exe_dir}/static")
            get_filename_component(_abs_static "${_exe_dir}/static" ABSOLUTE)
            set(_found "${_abs_static}")
            break()
        endif()
    endforeach()

    # 返回结果（可能为空）
    set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

# Helper: install_static_for_target(<target>)
# 如果在某个 exe 同级目录找到 static，则安装到 bin/<target>/static
function(install_static_for_target target)
    find_static_dir_for_target(${target} _static_dir)
    if(_static_dir)
        install(DIRECTORY "${_static_dir}/"
            DESTINATION "bin/${target}/static"
        )
        message(STATUS "[deploy] Installed static for ${target} from: ${_static_dir} -> <install_prefix>/bin/${target}/static")
    else()
        message(STATUS "[deploy] No static directory adjacent to built executable found for ${target}; skipping static install")
    endif()
endfunction()

# add_windows_runtime_deploy(target_name OUTPUT_DIR [WINDEPLOYQT <path-to-windeployqt>])
function(add_windows_runtime_deploy target_name output_dir)
    set(options)
    set(oneValueArgs WINDEPLOYQT)
    cmake_parse_arguments(_args "${options}" "${oneValueArgs}" "" ${ARGN})

    set(_TARGET "${target_name}")
    set(_OUT_DIR "${output_dir}")
    set(_WINDEPLOYQT "${_args_WINDEPLOYQT}")

    # 先尝试基于 exe 同级目录安装 static（如果存在）
    install_static_for_target(${_TARGET})

    # Windows-specific runtime deployment
    if(MINGW)
        find_program(PYTHON_EXECUTABLE python)
        find_program(OBJDUMP_EXECUTABLE objdump)

        if(OBJDUMP_EXECUTABLE)
            get_filename_component(_mingw_bin_dir "${OBJDUMP_EXECUTABLE}" DIRECTORY)
            get_filename_component(_mingw_root "${_mingw_bin_dir}" DIRECTORY)
        else()
            set(_mingw_root "")
        endif()

        if(_WINDEPLOYQT)
            install(CODE "
                message(STATUS \"[deploy] Running windeployqt for ${_TARGET} -> ${_OUT_DIR}\")
                if(NOT EXISTS \"${_OUT_DIR}/${_TARGET}.exe\")
                    message(WARNING \"Executable not found for windeployqt: ${_OUT_DIR}/${_TARGET}.exe\")
                else()
                    execute_process(
                        COMMAND \"${_WINDEPLOYQT}\" \"${_OUT_DIR}/${_TARGET}.exe\" --dir \"${_OUT_DIR}\"
                        RESULT_VARIABLE _wd_result
                        OUTPUT_VARIABLE _wd_out
                        ERROR_VARIABLE _wd_err
                    )
                    if(NOT _wd_result EQUAL 0)
                        message(WARNING \"windeployqt failed for ${_TARGET} with exit code \${_wd_result}\")
                        message(WARNING \"windeployqt stdout: \${_wd_out}\")
                        message(WARNING \"windeployqt stderr: \${_wd_err}\")
                    else()
                        message(STATUS \"windeployqt completed for ${_TARGET}\")
                    endif()
                endif()
            ")
        endif()

        if(NOT PYTHON_EXECUTABLE)
            message(WARNING "Python not found; cannot run mingw_collect_deps.py for ${_TARGET}")
        elseif(NOT OBJDUMP_EXECUTABLE)
            message(WARNING "objdump not found; cannot run mingw_collect_deps.py for ${_TARGET}")
        else()
            if(_mingw_root STREQUAL "")
                message(WARNING "Could not determine MinGW root; mingw_collect_deps.py may fail for ${_TARGET}")
            endif()

            install(CODE "
                message(STATUS \"[deploy] Collecting MinGW runtime dependencies for ${_TARGET} into ${_OUT_DIR}...\")
                if(NOT EXISTS \"${CMAKE_SOURCE_DIR}/scripts/mingw_collect_deps.py\")
                    message(WARNING \"MinGW dependency collection script not found at: ${CMAKE_SOURCE_DIR}/scripts/mingw_collect_deps.py\")
                else()
                    execute_process(
                        COMMAND \"${PYTHON_EXECUTABLE}\"
                                \"${CMAKE_SOURCE_DIR}/scripts/mingw_collect_deps.py\"
                                --mingw \"${_mingw_root}\"
                                --input \"${_OUT_DIR}/${_TARGET}.exe\"
                                --output \"${_OUT_DIR}\"
                        RESULT_VARIABLE _mc_result
                        OUTPUT_VARIABLE _mc_out
                        ERROR_VARIABLE _mc_err
                    )
                    if(NOT _mc_result EQUAL 0)
                        message(WARNING \"MinGW dependency collection for ${_TARGET} failed with exit code \${_mc_result}\")
                        message(WARNING \"stdout: \${_mc_out}\")
                        message(WARNING \"stderr: \${_mc_err}\")
                    else()
                        message(STATUS \"MinGW dependencies collected successfully for ${_TARGET}\")
                    endif()
                endif()
            ")
        endif()

    elseif(MSVC)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
            DESTINATION "bin/${_TARGET}"
            FILES_MATCHING
            PATTERN "*.dll"
            PATTERN "*.pdb" EXCLUDE
            PATTERN "vcpkg_installed" EXCLUDE
            PATTERN "Testing" EXCLUDE
            PATTERN "${_TARGET}_autogen" EXCLUDE
            PATTERN "CMakeFiles" EXCLUDE
            PATTERN ".qt" EXCLUDE
            PATTERN ".cmake" EXCLUDE
        )
        install(CODE "message(STATUS \"[deploy] MSVC/vcpkg DLLs will be installed for target: ${_TARGET} into <install_prefix>/bin/${_TARGET}\")")
    else()
        install(CODE "message(STATUS \"[deploy] add_windows_runtime_deploy: skipping Windows runtime steps for non-Windows platform for target ${_TARGET}\")")
    endif()
endfunction()
