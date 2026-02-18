// csr_interface.cpp - Memory-mapped CSR register access implementation
#include "driver/csr_interface.hpp"
#include "memory/address_map.hpp"

#include <fcntl.h>      // open()
#include <unistd.h>     // close()
#include <sys/mman.h>   // mmap(), munmap()
#include <cstring>      // strerror()
#include <cerrno>       // errno
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>

namespace accel {
namespace driver {

using namespace memory;

// =============================================================================
// Constructor / Destructor
// =============================================================================

CSRInterface::CSRInterface(uint32_t base_addr, size_t size)
    : fd_(-1)
    , mapped_base_(nullptr)
    , mapped_size_(size)
    , phys_base_(base_addr)
{
    // Open /dev/mem for physical memory access
    fd_ = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_ < 0) {
        throw CSRException("Failed to open /dev/mem: " + std::string(strerror(errno)) +
                          " (run as root or with sudo)");
    }
    
    // Map the CSR region into virtual address space
    mapped_base_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        fd_, base_addr);
    
    if (mapped_base_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        mapped_base_ = nullptr;
        throw CSRException("Failed to mmap CSR region at 0x" +
                          std::to_string(base_addr) + ": " + strerror(errno));
    }
}

CSRInterface::~CSRInterface() {
    if (mapped_base_ && mapped_base_ != MAP_FAILED) {
        munmap(const_cast<void*>(mapped_base_), mapped_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

// Move constructor
CSRInterface::CSRInterface(CSRInterface&& other) noexcept
    : fd_(other.fd_)
    , mapped_base_(other.mapped_base_)
    , mapped_size_(other.mapped_size_)
    , phys_base_(other.phys_base_)
{
    other.fd_ = -1;
    other.mapped_base_ = nullptr;
}

// Move assignment
CSRInterface& CSRInterface::operator=(CSRInterface&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        if (mapped_base_ && mapped_base_ != MAP_FAILED) {
            munmap(const_cast<void*>(mapped_base_), mapped_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        
        // Move from other
        fd_ = other.fd_;
        mapped_base_ = other.mapped_base_;
        mapped_size_ = other.mapped_size_;
        phys_base_ = other.phys_base_;
        
        other.fd_ = -1;
        other.mapped_base_ = nullptr;
    }
    return *this;
}

// =============================================================================
// Helper: Get pointer to register
// =============================================================================

volatile uint32_t* CSRInterface::regPtr(uint32_t offset) const {
    if (!mapped_base_) {
        throw CSRException("CSR interface not initialized");
    }
    if (offset >= mapped_size_) {
        throw CSRException("Register offset 0x" + std::to_string(offset) +
                          " out of range");
    }
    return reinterpret_cast<volatile uint32_t*>(
        static_cast<volatile uint8_t*>(mapped_base_) + offset
    );
}

// =============================================================================
// Core Register Operations
// =============================================================================

void CSRInterface::write32(uint32_t offset, uint32_t value) {
    *regPtr(offset) = value;
    // Memory barrier to ensure write completes
    __sync_synchronize();
}

uint32_t CSRInterface::read32(uint32_t offset) const {
    __sync_synchronize();
    return *regPtr(offset);
}

void CSRInterface::setBits(uint32_t offset, uint32_t mask) {
    uint32_t val = read32(offset);
    write32(offset, val | mask);
}

void CSRInterface::clearBits(uint32_t offset, uint32_t mask) {
    uint32_t val = read32(offset);
    write32(offset, val & ~mask);
}

bool CSRInterface::pollBits(uint32_t offset, uint32_t mask, uint32_t expected,
                            uint32_t timeout_us) const {
    auto start = std::chrono::steady_clock::now();
    auto timeout = std::chrono::microseconds(timeout_us);
    
    while (true) {
        uint32_t val = read32(offset);
        if ((val & mask) == expected) {
            return true;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            return false;
        }
        
        // Small sleep to avoid hammering the bus
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

// =============================================================================
// Convenience Methods
// =============================================================================

void CSRInterface::start() {
    // Write-1-to-pulse: write START bit to CTRL register
    write32(REG_CTRL, ctrl::START_BIT);
}

void CSRInterface::abort() {
    write32(REG_CTRL, ctrl::ABORT_BIT);
}

bool CSRInterface::isBusy() const {
    return (read32(REG_STATUS) & status::BUSY_BIT) != 0;
}

bool CSRInterface::waitDone(uint32_t timeout_us) const {
    // Poll until busy bit is cleared (not busy = done)
    return pollBits(REG_STATUS, status::BUSY_BIT, 0, timeout_us);
}

void CSRInterface::setIrqEnable(bool enable) {
    if (enable) {
        setBits(REG_CTRL, ctrl::IRQ_EN_BIT);
    } else {
        clearBits(REG_CTRL, ctrl::IRQ_EN_BIT);
    }
}


void CSRInterface::setMatrixDimensions(uint32_t M, uint32_t N, uint32_t K) {
    write32(REG_DIMS_M, M);
    write32(REG_DIMS_N, N);
    write32(REG_DIMS_K, K);
}

void CSRInterface::setTileDimensions(uint32_t MT, uint32_t NT, uint32_t KT) {
    write32(REG_DIMS_MT, MT);   // If this register exists
    write32(REG_DIMS_NT, NT);
    write32(REG_DIMS_KT, KT);
}

void CSRInterface::setScale(uint32_t scale_act, uint32_t scale_wgt) {
    write32(REG_SCALE_ACT, scale_act);
    write32(REG_SCALE_WGT, scale_wgt);
}

// =============================================================================
// DMA Operations
// =============================================================================

void CSRInterface::startWeightDMA(uint32_t src_addr, uint32_t len) {
    write32(REG_DMA_SRC_ADDR, src_addr);
    write32(REG_DMA_XFER_LEN, len);
    // Clear any previous done status, then start
    write32(REG_DMA_CTRL, dma_ctrl::DONE_BIT);  // W1C to clear done
    write32(REG_DMA_CTRL, dma_ctrl::START_BIT); // W1P to start
}

void CSRInterface::startActDMA(uint32_t src_addr, uint32_t len) {
    write32(REG_ACT_DMA_SRC, src_addr);
    write32(REG_ACT_DMA_LEN, len);
    write32(REG_ACT_DMA_CTRL, dma_ctrl::DONE_BIT);  // W1C to clear
    write32(REG_ACT_DMA_CTRL, dma_ctrl::START_BIT); // W1P to start
}

bool CSRInterface::waitWeightDMA(uint32_t timeout_us) const {
    // Wait for DONE bit to be set
    return pollBits(REG_DMA_CTRL, dma_ctrl::DONE_BIT, dma_ctrl::DONE_BIT, timeout_us);
}

bool CSRInterface::waitActDMA(uint32_t timeout_us) const {
    return pollBits(REG_ACT_DMA_CTRL, dma_ctrl::DONE_BIT, dma_ctrl::DONE_BIT, timeout_us);
}

// =============================================================================
// BSR Configuration
// =============================================================================

void CSRInterface::setBSRPointers(uint32_t ptr_addr, uint32_t idx_addr) {
    write32(REG_BSR_PTR_ADDR, ptr_addr);
    write32(REG_BSR_IDX_ADDR, idx_addr);
}

void CSRInterface::setBSRConfig(uint32_t num_blocks, uint32_t block_rows, uint32_t block_cols) {
    write32(REG_BSR_NUM_BLOCKS, num_blocks);
    write32(REG_BSR_BLOCK_ROWS, block_rows);
    write32(REG_BSR_BLOCK_COLS, block_cols);
    // Enable BSR mode (bit 0 of BSR_CONFIG)
    write32(REG_BSR_CONFIG, 0x00000101);  // Enable + version 1.0
}


// =============================================================================
// Performance Counters
// =============================================================================

CSRInterface::PerfCounters CSRInterface::readPerfCounters() const {
    PerfCounters pc;
    pc.total_cycles      = read32(REG_PERF_TOTAL);
    pc.active_cycles     = read32(REG_PERF_ACTIVE);
    pc.idle_cycles       = read32(REG_PERF_IDLE);
    pc.dma_bytes         = read32(REG_PERF_DMA_BYTES);
    pc.blocks_processed  = read32(REG_PERF_BLOCKS);
    pc.stall_cycles      = read32(REG_PERF_STALL);
    return pc;
}

// =============================================================================
// Debug
// =============================================================================

void CSRInterface::dumpRegisters() const {
    std::cout << "\n=== CSR Register Dump (Base: 0x" << std::hex << phys_base_ << ") ===\n";
    std::cout << std::setfill('0');
    
    auto printReg = [this](const char* name, uint32_t offset) {
        std::cout << "  " << std::setw(20) << std::left << name 
                  << "[0x" << std::hex << std::setw(2) << std::right << offset << "]: 0x"
                  << std::setw(8) << read32(offset) << std::dec << "\n";
    };
    
    std::cout << "-- Control/Status --\n";
    printReg("CTRL", REG_CTRL);
    printReg("STATUS", REG_STATUS);
    
    std::cout << "-- Dimensions --\n";
    printReg("DIMS_M", REG_DIMS_M);
    printReg("DIMS_N", REG_DIMS_N);
    printReg("DIMS_K", REG_DIMS_K);
    
    std::cout << "-- Performance --\n";
    printReg("PERF_TOTAL", REG_PERF_TOTAL);
    printReg("PERF_ACTIVE", REG_PERF_ACTIVE);
    printReg("PERF_IDLE", REG_PERF_IDLE);
    printReg("PERF_DMA_BYTES", REG_PERF_DMA_BYTES);
    printReg("PERF_BLOCKS", REG_PERF_BLOCKS);
    printReg("PERF_STALL", REG_PERF_STALL);
    
    std::cout << "-- DMA --\n";
    printReg("DMA_SRC_ADDR", REG_DMA_SRC_ADDR);
    printReg("DMA_XFER_LEN", REG_DMA_XFER_LEN);
    printReg("DMA_CTRL", REG_DMA_CTRL);
    printReg("ACT_DMA_SRC", REG_ACT_DMA_SRC);
    printReg("ACT_DMA_LEN", REG_ACT_DMA_LEN);
    printReg("ACT_DMA_CTRL", REG_ACT_DMA_CTRL);
    
    std::cout << "-- BSR --\n";
    printReg("BSR_CONFIG", REG_BSR_CONFIG);
    printReg("BSR_NUM_BLOCKS", REG_BSR_NUM_BLOCKS);
    printReg("BSR_BLOCK_ROWS", REG_BSR_BLOCK_ROWS);
    printReg("BSR_BLOCK_COLS", REG_BSR_BLOCK_COLS);
    printReg("BSR_PTR_ADDR", REG_BSR_PTR_ADDR);
    printReg("BSR_IDX_ADDR", REG_BSR_IDX_ADDR);
    printReg("BSR_STATUS", REG_BSR_STATUS);
    
    std::cout << "==========================================\n\n";
}



} // namespace driver
} // namespace accel
