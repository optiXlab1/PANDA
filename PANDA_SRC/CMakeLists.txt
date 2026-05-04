### make sure the user is doing an out-of-source build:
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed. use cmake -H. -Bbuild instead to build in the build folder")
endif()
###

cmake_minimum_required(VERSION 2.6)
project(panda C)

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    if(NOT MSVC)
        set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fprofile-arcs -ftest-coverage")
        set(LDFLAGS "-O0 -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
    endif()
endif()

# subdirectory with PANDA library
include_directories("${PROJECT_SOURCE_DIR}/PANDA")
add_subdirectory(PANDA)