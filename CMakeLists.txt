cmake_minimum_required(VERSION 3.9)
message(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

project(SIMDWrapper VERSION 1.0 LANGUAGES CXX)

include(GNUInstallDirs)

message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

# create cmakefile and install packages
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
#       INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
file(GLOB library_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
install(
    FILES ${library_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/SIMDWrapper
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)