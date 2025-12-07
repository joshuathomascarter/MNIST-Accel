cmake_minimum_required(VERSION 3.10)
project(verilator_tests)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(SOURCES test_mac8_verilator.cpp)

add_executable(runTests ${SOURCES})
target_link_libraries(runTests ${GTEST_LIBRARIES} pthread)