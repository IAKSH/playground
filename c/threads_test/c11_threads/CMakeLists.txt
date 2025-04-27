cmake_minimum_required(VERSION 3.10.0)
project(thread_test VERSION 0.1.0 LANGUAGES C)

include(CTest)
enable_testing()

file(GLOB C_FILES "*.c")

foreach(C_FILE ${C_FILES})
    get_filename_component(TARGET_NAME ${C_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${C_FILE})
    set_target_properties(${TARGET_NAME} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endforeach()

#add_test(NAME BasicThreads COMMAND basic_threads)
#add_test(NAME Mutex COMMAND mutex)
#add_test(NAME ConditionWithMutex COMMAND condition_with_mutex)
#add_test(NAME Detach COMMAND detach)
#add_test(NAME ThreadLocal COMMAND thread_local)
#add_test(NAME Misc COMMAND misc)
#add_test(NAME Atomic COMMAND atomic)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)