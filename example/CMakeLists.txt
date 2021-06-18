cmake_minimum_required(VERSION 3.9)

project(simd_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(SIMDWrapper)
set(CMAKE_BUILD_TYPE Release)

# AVX2
add_executable(${PROJECT_NAME}_AVX2 matrix.cpp)
target_link_libraries(${PROJECT_NAME}_AVX2 PRIVATE SIMDWrapper)
target_compile_options(${PROJECT_NAME}_AVX2 PRIVATE -mavx2 -mfma -O2)

# SSE4.2
add_executable(${PROJECT_NAME}_SSE matrix.cpp)
target_link_libraries(${PROJECT_NAME}_SSE PRIVATE SIMDWrapper)
target_compile_options(${PROJECT_NAME}_SSE PRIVATE -msse4.2 -mfma -O2)

# disable simd
add_executable(${PROJECT_NAME} matrix.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE SIMDWrapper)
target_compile_options(${PROJECT_NAME} PRIVATE -O2)
