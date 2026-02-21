# =============================================================================
# MNIST Accelerator C++ Driver
# 14×14 INT8 Weight-Stationary Systolic Array on PYNQ-Z2 (Zynq 7020)
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(mnist_accel VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DACCEL_LOG_LEVEL=0")
add_compile_options(-Wall -Wextra -Wpedantic)

# =============================================================================
# Include directories
# =============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# =============================================================================
# Source files
# =============================================================================

# Compute library sources
set(COMPUTE_SOURCES
    src/compute/tiling.cpp
    src/compute/bsr_encoder.cpp
    src/compute/golden_model.cpp
)

# Memory subsystem sources
set(MEMORY_SOURCES
    src/memory/buffer_manager.cpp
    src/memory/dma_controller.cpp
)

# Driver sources
set(DRIVER_SOURCES
    src/driver/csr_interface.cpp
    src/driver/accelerator.cpp
    src/driver/performance.cpp
)

# Utility sources
set(UTILS_SOURCES
    src/utils/npy_loader.cpp
    src/utils/logging.cpp
)

# All library sources
set(ALL_SOURCES
    ${COMPUTE_SOURCES}
    ${MEMORY_SOURCES}
    ${DRIVER_SOURCES}
    ${UTILS_SOURCES}
)

# =============================================================================
# Static library: libmnist_accel.a
# =============================================================================
add_library(mnist_accel_lib STATIC ${ALL_SOURCES})
target_include_directories(mnist_accel_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(mnist_accel_lib PROPERTIES OUTPUT_NAME "mnist_accel")

# =============================================================================
# Applications
# =============================================================================

# Main MNIST inference binary
add_executable(run_mnist_inference apps/run_mnist_inference.cpp)
target_link_libraries(run_mnist_inference PRIVATE mnist_accel_lib)

# Performance benchmarking tool
add_executable(benchmark apps/benchmark.cpp)
target_link_libraries(benchmark PRIVATE mnist_accel_lib)

# =============================================================================
# Tests
# =============================================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # -- test_tiling --
    add_executable(test_tiling tests/test_tiling.cpp)
    target_link_libraries(test_tiling PRIVATE mnist_accel_lib)
    add_test(NAME test_tiling COMMAND test_tiling)

    # -- test_bsr_encoder --
    add_executable(test_bsr_encoder tests/test_bsr_encoder.cpp)
    target_link_libraries(test_bsr_encoder PRIVATE mnist_accel_lib)
    add_test(NAME test_bsr_encoder COMMAND test_bsr_encoder)

    # -- test_buffer_manager --
    add_executable(test_buffer_manager tests/test_buffer_manager.cpp)
    target_link_libraries(test_buffer_manager PRIVATE mnist_accel_lib)
    add_test(NAME test_buffer_manager COMMAND test_buffer_manager)

    # -- test_dma --
    add_executable(test_dma tests/test_dma.cpp)
    target_link_libraries(test_dma PRIVATE mnist_accel_lib)
    add_test(NAME test_dma COMMAND test_dma)

    # -- test_end_to_end --
    add_executable(test_end_to_end tests/test_end_to_end.cpp)
    target_link_libraries(test_end_to_end PRIVATE mnist_accel_lib)
    add_test(NAME test_end_to_end COMMAND test_end_to_end)
endif()

# =============================================================================
# Cross-compilation support (PYNQ-Z2 / ARM Cortex-A9)
# =============================================================================
# To cross-compile for PYNQ-Z2, use:
#   cmake -DCMAKE_TOOLCHAIN_FILE=<path>/arm-linux-gnueabihf.cmake ..
#
# Or set the compiler directly:
#   cmake -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
#         -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ ..

# =============================================================================
# Install rules
# =============================================================================
install(TARGETS run_mnist_inference benchmark
        RUNTIME DESTINATION bin)
install(TARGETS mnist_accel_lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/
        DESTINATION include/mnist_accel)

# =============================================================================
# Print summary
# =============================================================================
message(STATUS "=== MNIST Accelerator C++ Driver ===")
message(STATUS "Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "Build tests:     ${BUILD_TESTS}")
message(STATUS "Install prefix:  ${CMAKE_INSTALL_PREFIX}")
