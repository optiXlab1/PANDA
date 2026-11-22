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

include(CTest)
if(BUILD_TESTING)
    add_executable(test_initialization tests/test_initialization.c PANDA/function_evaluator.c)
    target_link_libraries(test_initialization PANDA_fake)
    if(NOT MSVC)
        target_link_libraries(test_initialization m)
    endif()
    add_test(NAME initialization COMMAND test_initialization)

    add_executable(test_lbfgs_scale_guard tests/test_lbfgs_scale_guard.c)
    target_link_libraries(test_lbfgs_scale_guard PANDA_fake)
    if(NOT MSVC)
        target_link_libraries(test_lbfgs_scale_guard m)
    endif()
    add_test(NAME lbfgs_scale_guard COMMAND test_lbfgs_scale_guard)
endif()
