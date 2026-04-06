// dma_controller.hpp - DMA transfer management for PL-PS data movement
// Manages three hardware DMA engines: weight (bsr_dma), activation (act_dma),
// and output (out_dma). Handles DDR buffer allocation, double-buffering,
// and transfer sequencing.
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>

namespace accel {

// Forward declaration — avoids circular include with csr_interface.hpp
namespace driver { class CSRInterface; }

namespace memory {

// =============================================================================
// DMAChannel — Identifies which hardware DMA engine to target
// =============================================================================
enum class DMAChannel {
    Weight,      // bsr_dma.sv  — loads BSR weights + row_ptr + col_idx
    Activation,  // act_dma.sv  — loads activation tile
    Output       // out_dma.sv  — writes output accumulators to DDR
};

// =============================================================================
// TransferDescriptor — Describes a single DMA transfer
// =============================================================================
struct TransferDescriptor {
    DMAChannel channel;        // Which DMA engine
    uint32_t   src_addr;       // DDR source address (for reads) or 0
    uint32_t   dst_addr;       // DDR destination address (for writes) or 0
    uint32_t   length_bytes;   // Transfer size in bytes
    uint32_t   bank;           // Target buffer bank (0 or 1 for double-buffer)
};

// =============================================================================
// TransferStats — Statistics for a completed transfer
// =============================================================================
struct TransferStats {
    uint32_t bytes_transferred;
    uint32_t elapsed_us;       // Approximate wall-clock time
    double   bandwidth_mbps;   // Measured bandwidth in MB/s
};

// =============================================================================
// DMAController — High-level DMA manager
// =============================================================================
//
// This class sits one layer above CSRInterface.  CSRInterface knows how to
// write individual registers; DMAController knows the *protocol*:
//   1. Copy data from user buffer into the DDR reserved region.
//   2. Program the correct CSR registers (src_addr, length, bank).
//   3. Pulse the START bit.
//   4. Poll until DONE.
//   5. Return statistics.
//
// It also manages the DDR reserved region (PL_DDR_OFFSET) so higher layers
// don't have to compute absolute addresses.
//
class DMAController {
public:
    // -------------------------------------------------------------------------
    // Construction / Destruction
    // -------------------------------------------------------------------------

    // @param csr       Reference to an already-initialised CSRInterface
    // @param ddr_phys  Physical base of the accelerator DDR region
    //                  (default: DDR_BASE_ADDR + PL_DDR_OFFSET = 0x10000000)
    explicit DMAController(driver::CSRInterface& csr,
                           uint32_t ddr_phys = 0x10000000);
    ~DMAController();

    // Non-copyable (references CSR + owns mmap)
    DMAController(const DMAController&) = delete;
    DMAController& operator=(const DMAController&) = delete;

    // Movable
    DMAController(DMAController&& other) noexcept;
    DMAController& operator=(DMAController&& other) noexcept;

    // -------------------------------------------------------------------------
    // DDR Buffer Management
    // -------------------------------------------------------------------------

    // Map the accelerator DDR region into user-space (via /dev/mem).
    // Must be called before any load/store helpers.
    void mapDDR(size_t size = 0x00800000);  // default 8 MB

    // Absolute DDR addresses for each data region
    uint32_t weightsAddr()  const;
    uint32_t actsAddr()     const;
    uint32_t bsrPtrAddr()   const;
    uint32_t bsrIdxAddr()   const;
    uint32_t outputAddr()   const;

    // -------------------------------------------------------------------------
    // Data Loading — copy from host memory into DDR reserved region
    // -------------------------------------------------------------------------

    // Copy raw weight block data into the DDR weight region
    void loadWeights(const void* data, size_t bytes);

    // Copy activation tile data into the DDR activation region
    void loadActivations(const void* data, size_t bytes);

    // Copy BSR metadata (row_ptr array) into DDR
    void loadBSRRowPtr(const void* data, size_t bytes);

    // Copy BSR metadata (col_idx array) into DDR
    void loadBSRColIdx(const void* data, size_t bytes);

    // Read output results from DDR output region
    void readOutput(void* dst, size_t bytes) const;

    // -------------------------------------------------------------------------
    // DMA Transfer Control
    // -------------------------------------------------------------------------

    // Start a weight DMA transfer (DDR → weight BRAM)
    // @param bytes  Number of bytes to transfer (0 = use full weight region)
    void startWeightTransfer(uint32_t bytes = 0);

    // Start an activation DMA transfer (DDR → activation BRAM)
    // @param bytes  Number of bytes to transfer (0 = use full act region)
    void startActivationTransfer(uint32_t bytes = 0);

    // Start output DMA transfer (output accumulator → DDR)
    // Note: out_dma.sv auto-triggers on sched_done if dst_addr != 0.
    // This method just sets the destination address.
    void configureOutputDMA();

    // Execute a generic transfer descriptor
    TransferStats executeTransfer(const TransferDescriptor& desc);

    // -------------------------------------------------------------------------
    // Synchronisation
    // -------------------------------------------------------------------------

    // Wait for weight DMA to complete
    bool waitWeightDone(uint32_t timeout_us = 5000000) const;

    // Wait for activation DMA to complete
    bool waitActivationDone(uint32_t timeout_us = 5000000) const;

    // Wait for ALL pending DMA transfers to complete
    bool waitAll(uint32_t timeout_us = 10000000) const;

    // Check if any DMA channel is busy
    bool isAnyBusy() const;

    // -------------------------------------------------------------------------
    // Double-Buffering Support
    // -------------------------------------------------------------------------

    // Select which buffer bank DMA writes into (0 or 1)
    void setWeightBank(uint32_t bank);
    void setActivationBank(uint32_t bank);

    // Get the current write-bank for each channel
    uint32_t weightBank() const { return wgt_bank_; }
    uint32_t activationBank() const { return act_bank_; }

    // Swap both banks in one call (for ping-pong)
    void swapBanks();

    // -------------------------------------------------------------------------
    // Statistics / Debug
    // -------------------------------------------------------------------------

    // Total bytes transferred since construction
    uint64_t totalBytesTransferred() const { return total_bytes_; }

    // Number of completed transfers
    uint32_t transferCount() const { return transfer_count_; }

    // Print a summary of the DDR layout and current state
    void dumpState() const;

private:
    driver::CSRInterface& csr_;  // Reference to register interface
    uint32_t ddr_phys_;          // Physical base of accel DDR region

    // mmap'd pointer to the DDR reserved region (for load/store helpers)
    int          ddr_fd_;
    volatile void* ddr_mapped_;
    size_t       ddr_mapped_size_;

    // Double-buffer bank selection
    uint32_t wgt_bank_;
    uint32_t act_bank_;

    // Statistics
    uint64_t total_bytes_;
    uint32_t transfer_count_;

    // Internal helpers
    void copyToDDR(uint32_t offset, const void* src, size_t bytes);
    void copyFromDDR(uint32_t offset, void* dst, size_t bytes) const;
    volatile void* ddrPtr(uint32_t offset) const;
};

// =============================================================================
// Exception for DMA errors
// =============================================================================
class DMAException : public std::runtime_error {
public:
    explicit DMAException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace memory
} // namespace accel
