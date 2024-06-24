cmake_minimum_required(VERSION 3.0.0)
project(crow-rest-api-test VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

message(OPENSSL_CRYPTO_LIBRARY = ${OPENSSL_CRYPTO_LIBRARY})
message(OPENSSL_SSL_LIBRARY = ${OPENSSL_SSL_LIBRARY})

add_executable(crow-rest-api-test)
target_include_directories(crow-rest-api-test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/)
target_sources(crow-rest-api-test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/source/main.cpp)
target_link_libraries(crow-rest-api-test PUBLIC ${OPENSSL_CRYPTO_LIBRARY} ${OPENSSL_SSL_LIBRARY} z)

add_executable(blueprints-test)
target_include_directories(blueprints-test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/)
target_sources(blueprints-test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/source/blueprints_test.cpp)

find_library(PQXX_LIB pqxx)
find_library(PQ_LIB pq)

message(PQXX_LIB = ${PQXX_LIB})
message(PQ_LIB = ${PQ_LIB})

add_executable(pqxx-test)
target_include_directories(pqxx-test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/)
target_sources(pqxx-test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/source/pqxx_test.cpp)
target_link_libraries(pqxx-test PUBLIC ${PQXX_LIB} ${PQ_LIB})

execute_process( COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/templates ${CMAKE_CURRENT_LIST_DIR}/out/build/linux-gcc-11/templates)
execute_process( COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/static ${CMAKE_CURRENT_LIST_DIR}/out/build/linux-gcc-11/static)
execute_process( COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/tls ${CMAKE_CURRENT_LIST_DIR}/out/build/linux-gcc-11/tls)