// csr_interface.hpp - CSR register read/write interface via mmap
// Provides memory-mapped I/O access to accelerator CSR registers
#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>

namespace accel {
namespace driver {

// =============================================================================
// CSRInterface - Memory-mapped register access for Pynq-Z2
// =============================================================================
class CSRInterface {
public:
    // -------------------------------------------------------------------------
    // Constructor: Opens /dev/mem and maps the CSR region
    // @param base_addr: Physical base address of accelerator (default: 0x43C00000)
    // @param size: Size of mapped region in bytes (default: 4KB = 0x1000)
    // -------------------------------------------------------------------------
    explicit CSRInterface(uint32_t base_addr = 0x43C00000, size_t size = 0x1000);
    
    // Destructor: Unmaps memory and closes file descriptor
    ~CSRInterface();
    
    // Non-copyable (owns mmap resource)
    CSRInterface(const CSRInterface&) = delete;
    CSRInterface& operator=(const CSRInterface&) = delete;
    
    // Movable
    CSRInterface(CSRInterface&& other) noexcept;
    CSRInterface& operator=(CSRInterface&& other) noexcept;
    
    // -------------------------------------------------------------------------
    // Core Register Operations
    // -------------------------------------------------------------------------
    
    // Write 32-bit value to register at offset
    void write32(uint32_t offset, uint32_t value);
    
    // Read 32-bit value from register at offset
    uint32_t read32(uint32_t offset) const;
    
    // Read-modify-write: set specific bits
    void setBits(uint32_t offset, uint32_t mask);
    
    // Read-modify-write: clear specific bits
    void clearBits(uint32_t offset, uint32_t mask);
    
    // Poll register until (reg & mask) == expected, with timeout
    // Returns true if condition met, false on timeout
    bool pollBits(uint32_t offset, uint32_t mask, uint32_t expected,
                  uint32_t timeout_us = 1000000) const;
    
    // -------------------------------------------------------------------------
    // Convenience Methods (built on top of address_map.hpp constants)
    // -------------------------------------------------------------------------
    
    // Start accelerator execution (W1P to CTRL[0])
    void start();
    
    // Abort current operation (W1P to CTRL[1])
    void abort();
    
    // Check if accelerator is busy
    bool isBusy() const;
    
    // Wait for accelerator to complete (polls STATUS[0])
    bool waitDone(uint32_t timeout_us = 10000000) const;
    
    // Enable/disable interrupt
    void setIrqEnable(bool enable);
    
    // -------------------------------------------------------------------------
    // DMA Operations
    // -------------------------------------------------------------------------
    
    // Start weight DMA transfer
    void startWeightDMA(uint32_t src_addr, uint32_t len);
    
    // Start activation DMA transfer
    void startActDMA(uint32_t src_addr, uint32_t len);
    
    // Wait for weight DMA to complete
    bool waitWeightDMA(uint32_t timeout_us = 1000000) const;
    
    // Wait for activation DMA to complete
    bool waitActDMA(uint32_t timeout_us = 1000000) const;
    
    // -------------------------------------------------------------------------
    // BSR Configuration
    // -------------------------------------------------------------------------
    
    // Configure BSR metadata addresses
    void setBSRPointers(uint32_t ptr_addr, uint32_t idx_addr);
    
    // Set BSR block configuration
    void setBSRConfig(uint32_t num_blocks, uint32_t block_rows, uint32_t block_cols);
    
    // -------------------------------------------------------------------------
    // Performance Counters
    // -------------------------------------------------------------------------
    struct PerfCounters {
        uint32_t total_cycles;
        uint32_t active_cycles;
        uint32_t idle_cycles;
        uint32_t dma_bytes;
        uint32_t blocks_processed;
        uint32_t stall_cycles;
    };
    
    PerfCounters readPerfCounters() const;
    
    // -------------------------------------------------------------------------
    // Debug
    // -------------------------------------------------------------------------
    
    // Dump all CSR registers to stdout
    void dumpRegisters() const;
    
    // Get mapped virtual address (for advanced use)
    volatile void* getMappedAddr() const { return mapped_base_; }
    
    // Check if successfully mapped
    bool isValid() const { return mapped_base_ != nullptr; }
    
    // -------------------------------------------------------------------------
    // Matrix/Tiling Configuration
    // -------------------------------------------------------------------------
    
    void setMatrixDimensions(uint32_t M, uint32_t N, uint32_t K);
    void setTileDimensions(uint32_t MT, uint32_t NT, uint32_t KT);
    void setScale(uint32_t scale_act, uint32_t scale_wgt);

private:
    int fd_;                      // /dev/mem file descriptor
    volatile void* mapped_base_;  // mmap'd virtual address
    size_t mapped_size_;          // Size of mapped region
    uint32_t phys_base_;          // Physical base address (for debug)
    
    // Helper to get pointer to specific offset
    volatile uint32_t* regPtr(uint32_t offset) const;
};

// =============================================================================
// Exception for CSR errors
// =============================================================================
class CSRException : public std::runtime_error {
public:
    explicit CSRException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace driver
} // namespace accel
