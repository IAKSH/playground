cmake_minimum_required(VERSION 3.10.0)
project(ctest_intro VERSION 0.1.0 LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.10)
project(MyCppProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_executable(addition_test addition_test.cpp)
add_executable(pass_test pass_test.cpp)
add_executable(wrong_test wrong_test.cpp)
add_executable(test_with_args test_with_args.cpp)
add_executable(long_running_test long_running_test.cpp)

include(CTest)
enable_testing()

add_test(NAME AdditionTest COMMAND addition_test)
add_test(NAME PassTest COMMAND pass_test)
add_test(NAME WrongTest COMMAND wrong_test)
add_test(NAME TestWithArgs_WithTrueArgs COMMAND test_with_args pass)
add_test(NAME TestWithArgs_WithFalseArgs COMMAND test_with_args fail)
add_test(NAME TestWithArgs_Without COMMAND test_with_args)

# 设置预期输出
add_test(NAME AdditionTest1 COMMAND addition_test)
set_tests_properties(AdditionTest1 PROPERTIES PASS_REGULAR_EXPRESSION "= 5")

# 设置测试超时时间为 5 秒
add_test(NAME TimeTest COMMAND long_running_test)
set_tests_properties(TimeTest PROPERTIES TIMEOUT 5)

# 使用set_tests_properties设置的，可以混在一起用
# 还有其他的属性，例如：

# 依赖其他测试，依赖的测试执行后才会执行本测试
# set_tests_properties(my_test PROPERTIES DEPENDS another_test)
# 设置测试为顺序运行
# set_tests_properties(my_test PROPERTIES RUN_SERIAL ON)
# 指定测试所需文件，如果没有则不会运行该测试
# set_tests_properties(my_test PROPERTIES REQUIRED_FILES "input.txt")
# 为测试添加标签，便于管理，可以通过标签运行特定测试集
# set_tests_properties(my_test PROPERTIES LABELS "unit;performance")
# 为测试指定环境变量
# set_tests_properties(my_test PROPERTIES ENVIRONMENT "MY_VAR=123")
# 如果测试返回指定代码，则视为已跳过该测试
# set_tests_properties(my_test PROPERTIES SKIP_RETURN_CODE 127)
# 指定测试预期失败，失败结果会被标记为通过
# set_tests_properties(my_test PROPERTIES WILL_FAIL ON)

# 多种属性可以一并设置，例如：
# set_tests_properties(my_test PROPERTIES
#     TIMEOUT 5
#     PASS_REGULAR_EXPRESSION "Success"
#     FAIL_REGULAR_EXPRESSION "Failure"
#     DEPENDS another_test
#     RUN_SERIAL ON
#     LABELS "integration"
#     ENVIRONMENT "TEST_ENV=prod"
# )

# 另外，似乎还有两种用正则表达式来判断是否通过的：
# set_tests_properties(my_test PROPERTIES PASS_REGULAR_EXPRESSION "Expected output: Success")
# set_tests_properties(my_test PROPERTIES FAIL_REGULAR_EXPRESSION "Error: Failure detected")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
