cmake_minimum_required(VERSION 3.9)

project(simd_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(SIMDWrapper)
add_executable(${PROJECT_NAME} matrix.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE SIMDWrapper)