# CMakeLists.txt for MNIST-Accel SoC Project
# Handles RTL simulation with cocotb/Verilator

cmake_minimum_required(VERSION 3.16)
project(mnist_accel_soc)

# Toolchain configuration
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Simulation tools
find_program(VERILATOR verilator)
find_program(COCOTB_CONFIG cocotb-config)

if(NOT VERILATOR OR NOT COCOTB_CONFIG)
    message(WARNING "Verilator or cocotb not found - simulation will not work")
endif()

# Project structure
set(HW_RTL_DIR ${CMAKE_SOURCE_DIR}/hw/rtl)
set(HW_SIM_DIR ${CMAKE_SOURCE_DIR}/hw/sim)
set(FW_DIR ${CMAKE_SOURCE_DIR}/fw)

# Source files for simulation
set(RTL_SOURCES
    ${HW_RTL_DIR}/top/soc_pkg.sv
    ${HW_RTL_DIR}/top/soc_top.sv
    ${HW_RTL_DIR}/top/simple_cpu.sv
    ${HW_RTL_DIR}/top/obi_to_axi.sv
    ${HW_RTL_DIR}/top/axi_crossbar.sv
    ${HW_RTL_DIR}/memory/boot_rom.sv
    ${HW_RTL_DIR}/memory/sram_ctrl.sv
    ${HW_RTL_DIR}/periph/uart_ctrl.sv
    ${HW_RTL_DIR}/periph/timer_ctrl.sv
    ${HW_RTL_DIR}/periph/gpio_ctrl.sv
    ${HW_RTL_DIR}/periph/plic.sv
)

# Firmware build
add_custom_target(firmware
    COMMAND cd ${FW_DIR} && make all
    OUTPUT ${FW_DIR}/firmware.hex
    COMMENT "Building RISC-V firmware"
)

# Simulation targets will be added via cocotb integration
message(STATUS "CMake configuration complete")
message(STATUS "To run simulation: cd hw/sim && make cocotb")
message(STATUS "To build firmware: cd fw && make")
