// dma_controller.cpp - DMA transfer management implementation
// Manages weight, activation, and output DMA engines.
// Handles DDR buffer mapping, data staging, and transfer sequencing.
#include "memory/dma_controller.hpp"
#include "memory/address_map.hpp"
#include "driver/csr_interface.hpp"

#include <fcntl.h>       // open()
#include <unistd.h>      // close()
#include <sys/mman.h>    // mmap(), munmap()
#include <cstring>       // memcpy(), strerror()
#include <cerrno>        // errno
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cassert>

namespace accel {
namespace memory {

using namespace accel::memory;  // address_map constants

// =============================================================================
// Constructor / Destructor
// =============================================================================

DMAController::DMAController(driver::CSRInterface& csr, uint32_t ddr_phys)
    : csr_(csr)
    , ddr_phys_(ddr_phys)
    , ddr_fd_(-1)
    , ddr_mapped_(nullptr)
    , ddr_mapped_size_(0)
    , wgt_bank_(0)
    , act_bank_(0)
    , total_bytes_(0)
    , transfer_count_(0)
{
}

DMAController::~DMAController() {
    // Unmap DDR region if mapped
    if (ddr_mapped_ && ddr_mapped_ != MAP_FAILED) {
        munmap(const_cast<void*>(ddr_mapped_), ddr_mapped_size_);
    }
    if (ddr_fd_ >= 0) {
        close(ddr_fd_);
    }
}

// Move constructor
DMAController::DMAController(DMAController&& other) noexcept
    : csr_(other.csr_)
    , ddr_phys_(other.ddr_phys_)
    , ddr_fd_(other.ddr_fd_)
    , ddr_mapped_(other.ddr_mapped_)
    , ddr_mapped_size_(other.ddr_mapped_size_)
    , wgt_bank_(other.wgt_bank_)
    , act_bank_(other.act_bank_)
    , total_bytes_(other.total_bytes_)
    , transfer_count_(other.transfer_count_)
{
    // Invalidate the source so it doesn't close/unmap our resources
    other.ddr_fd_ = -1;
    other.ddr_mapped_ = nullptr;
}

// Move assignment
DMAController& DMAController::operator=(DMAController&& other) noexcept {
    if (this != &other) {
        // Clean up our current resources
        if (ddr_mapped_ && ddr_mapped_ != MAP_FAILED) {
            munmap(const_cast<void*>(ddr_mapped_), ddr_mapped_size_);
        }
        if (ddr_fd_ >= 0) {
            close(ddr_fd_);
        }

        // Take ownership of other's resources
        // Note: csr_ is a reference, can't reassign — must be same target
        ddr_phys_       = other.ddr_phys_;
        ddr_fd_         = other.ddr_fd_;
        ddr_mapped_     = other.ddr_mapped_;
        ddr_mapped_size_= other.ddr_mapped_size_;
        wgt_bank_       = other.wgt_bank_;
        act_bank_       = other.act_bank_;
        total_bytes_    = other.total_bytes_;
        transfer_count_ = other.transfer_count_;

        // Invalidate other
        other.ddr_fd_ = -1;
        other.ddr_mapped_ = nullptr;
    }
    return *this;
}

// =============================================================================
// DDR Buffer Management
// =============================================================================

void DMAController::mapDDR(size_t size) {
    if (ddr_mapped_) {
        return;  // Already mapped
    }

    ddr_fd_ = open("/dev/mem", O_RDWR | O_SYNC);
    if (ddr_fd_ < 0) {
        throw DMAException("Failed to open /dev/mem for DDR mapping: " +
                          std::string(strerror(errno)));
    }

    ddr_mapped_ = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, ddr_fd_, ddr_phys_);

    if (ddr_mapped_ == MAP_FAILED) {
        close(ddr_fd_);
        ddr_fd_ = -1;
        ddr_mapped_ = nullptr;
        throw DMAException("Failed to mmap DDR region at 0x" +
                          std::to_string(ddr_phys_) + ": " + strerror(errno));
    }

    ddr_mapped_size_ = size;
}

// Compute absolute DDR addresses for each data region
uint32_t DMAController::weightsAddr() const {
    return ddr_phys_ + ddr_layout::WEIGHTS_OFFSET;
}

uint32_t DMAController::actsAddr() const {
    return ddr_phys_ + ddr_layout::ACTS_OFFSET;
}

uint32_t DMAController::bsrPtrAddr() const {
    return ddr_phys_ + ddr_layout::BSR_PTR_OFFSET;
}

uint32_t DMAController::bsrIdxAddr() const {
    return ddr_phys_ + ddr_layout::BSR_IDX_OFFSET;
}

uint32_t DMAController::outputAddr() const {
    return ddr_phys_ + ddr_layout::OUTPUT_OFFSET;
}

// =============================================================================
// Internal DDR Copy Helpers
// =============================================================================

volatile void* DMAController::ddrPtr(uint32_t offset) const {
    if (!ddr_mapped_) {
        throw DMAException("DDR not mapped — call mapDDR() first");
    }
    if (offset >= ddr_mapped_size_) {
        throw DMAException("DDR offset 0x" + std::to_string(offset) +
                          " exceeds mapped size");
    }
    return reinterpret_cast<volatile void*>(
        static_cast<volatile uint8_t*>(ddr_mapped_) + offset
    );
}

void DMAController::copyToDDR(uint32_t offset, const void* src, size_t bytes) {
    volatile void* dst = ddrPtr(offset);
    // memcpy is safe here because we're writing to mmap'd device memory
    memcpy(const_cast<void*>(dst), src, bytes);
    // Memory barrier to ensure all writes visible to hardware
    __sync_synchronize();
}

void DMAController::copyFromDDR(uint32_t offset, void* dst, size_t bytes) const {
    __sync_synchronize();
    volatile void* src = ddrPtr(offset);
    memcpy(dst, const_cast<void*>(src), bytes);
}

// =============================================================================
// Data Loading — Copy from Host Buffer to DDR Reserved Region
// =============================================================================

void DMAController::loadWeights(const void* data, size_t bytes) {
    if (bytes > ddr_layout::WEIGHTS_SIZE) {
        throw DMAException("Weight data (" + std::to_string(bytes) +
                          " bytes) exceeds allocated region (" +
                          std::to_string(ddr_layout::WEIGHTS_SIZE) + " bytes)");
    }
    copyToDDR(ddr_layout::WEIGHTS_OFFSET, data, bytes);
}

void DMAController::loadActivations(const void* data, size_t bytes) {
    if (bytes > ddr_layout::ACTS_SIZE) {
        throw DMAException("Activation data (" + std::to_string(bytes) +
                          " bytes) exceeds allocated region (" +
                          std::to_string(ddr_layout::ACTS_SIZE) + " bytes)");
    }
    copyToDDR(ddr_layout::ACTS_OFFSET, data, bytes);
}

void DMAController::loadBSRRowPtr(const void* data, size_t bytes) {
    if (bytes > ddr_layout::BSR_PTR_SIZE) {
        throw DMAException("BSR row_ptr data exceeds allocated region");
    }
    copyToDDR(ddr_layout::BSR_PTR_OFFSET, data, bytes);
}

void DMAController::loadBSRColIdx(const void* data, size_t bytes) {
    if (bytes > ddr_layout::BSR_IDX_SIZE) {
        throw DMAException("BSR col_idx data exceeds allocated region");
    }
    copyToDDR(ddr_layout::BSR_IDX_OFFSET, data, bytes);
}

void DMAController::readOutput(void* dst, size_t bytes) const {
    if (bytes > ddr_layout::OUTPUT_SIZE) {
        throw DMAException("Output read size exceeds allocated region");
    }
    copyFromDDR(ddr_layout::OUTPUT_OFFSET, dst, bytes);
}

// =============================================================================
// DMA Transfer Control
// =============================================================================

void DMAController::startWeightTransfer(uint32_t bytes) {
    if (bytes == 0) bytes = ddr_layout::WEIGHTS_SIZE;

    // Program CSR registers: tell bsr_dma where to read from
    csr_.startWeightDMA(weightsAddr(), bytes);
}

void DMAController::startActivationTransfer(uint32_t bytes) {
    if (bytes == 0) bytes = ddr_layout::ACTS_SIZE;

    // Program CSR registers: tell act_dma where to read from
    csr_.startActDMA(actsAddr(), bytes);
}

void DMAController::configureOutputDMA() {
    // out_dma.sv auto-triggers on sched_done when dst_addr != 0.
    // We just set the destination address in the CSR DMA_DST register.
    csr_.write32(REG_DMA_DST_ADDR, outputAddr());
}

TransferStats DMAController::executeTransfer(const TransferDescriptor& desc) {
    auto t_start = std::chrono::steady_clock::now();

    switch (desc.channel) {
    case DMAChannel::Weight:
        setWeightBank(desc.bank);
        csr_.startWeightDMA(desc.src_addr, desc.length_bytes);
        if (!waitWeightDone()) {
            throw DMAException("Weight DMA transfer timed out");
        }
        break;

    case DMAChannel::Activation:
        setActivationBank(desc.bank);
        csr_.startActDMA(desc.src_addr, desc.length_bytes);
        if (!waitActivationDone()) {
            throw DMAException("Activation DMA transfer timed out");
        }
        break;

    case DMAChannel::Output:
        // Output DMA is auto-triggered; configure dst_addr
        csr_.write32(REG_DMA_DST_ADDR, desc.dst_addr);
        // Wait handled externally (sched_done triggers it)
        break;
    }

    auto t_end = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          t_end - t_start).count();

    // Update statistics
    total_bytes_ += desc.length_bytes;
    transfer_count_++;

    TransferStats stats;
    stats.bytes_transferred = desc.length_bytes;
    stats.elapsed_us        = static_cast<uint32_t>(elapsed_us);
    stats.bandwidth_mbps    = (elapsed_us > 0)
        ? (static_cast<double>(desc.length_bytes) / elapsed_us)  // bytes/µs = MB/s
        : 0.0;

    return stats;
}

// =============================================================================
// Synchronisation
// =============================================================================

bool DMAController::waitWeightDone(uint32_t timeout_us) const {
    return csr_.waitWeightDMA(timeout_us);
}

bool DMAController::waitActivationDone(uint32_t timeout_us) const {
    return csr_.waitActDMA(timeout_us);
}

bool DMAController::waitAll(uint32_t timeout_us) const {
    // Wait for both input DMA engines
    bool wgt_ok = waitWeightDone(timeout_us);
    bool act_ok = waitActivationDone(timeout_us);
    return wgt_ok && act_ok;
}

bool DMAController::isAnyBusy() const {
    uint32_t dma_status = csr_.read32(REG_DMA_CTRL);
    uint32_t act_status = csr_.read32(REG_ACT_DMA_CTRL);
    return (dma_status & dma_ctrl::BUSY_BIT) ||
           (act_status & dma_ctrl::BUSY_BIT);
}

// =============================================================================
// Double-Buffering Support
// =============================================================================

void DMAController::setWeightBank(uint32_t bank) {
    wgt_bank_ = bank & 1;  // Clamp to 0 or 1
    // REG_BUFF[1] controls weight write bank
    uint32_t buff_val = csr_.read32(REG_BUFF);
    if (wgt_bank_) {
        buff_val |=  (1 << 1);  // Set bit 1
    } else {
        buff_val &= ~(1 << 1);  // Clear bit 1
    }
    csr_.write32(REG_BUFF, buff_val);
}

void DMAController::setActivationBank(uint32_t bank) {
    act_bank_ = bank & 1;
    // REG_BUFF[0] controls activation write bank
    uint32_t buff_val = csr_.read32(REG_BUFF);
    if (act_bank_) {
        buff_val |=  (1 << 0);  // Set bit 0
    } else {
        buff_val &= ~(1 << 0);  // Clear bit 0
    }
    csr_.write32(REG_BUFF, buff_val);
}

void DMAController::swapBanks() {
    // Toggle both banks: 0→1 or 1→0
    setWeightBank(wgt_bank_ ^ 1);
    setActivationBank(act_bank_ ^ 1);
}

// =============================================================================
// Debug
// =============================================================================

void DMAController::dumpState() const {
    std::cout << "\n=== DMA Controller State ===\n";
    std::cout << std::hex << std::setfill('0');

    std::cout << "DDR Physical Base:  0x" << std::setw(8) << ddr_phys_ << "\n";
    std::cout << "DDR Mapped:         " << (ddr_mapped_ ? "YES" : "NO") << "\n";
    std::cout << "DDR Mapped Size:    " << std::dec << ddr_mapped_size_ << " bytes\n";

    std::cout << std::hex;
    std::cout << "\n-- DDR Region Addresses --\n";
    std::cout << "  Weights:    0x" << std::setw(8) << weightsAddr()
              << " (" << std::dec << ddr_layout::WEIGHTS_SIZE << " bytes)\n";
    std::cout << std::hex;
    std::cout << "  Activations:0x" << std::setw(8) << actsAddr()
              << " (" << std::dec << ddr_layout::ACTS_SIZE << " bytes)\n";
    std::cout << std::hex;
    std::cout << "  BSR row_ptr:0x" << std::setw(8) << bsrPtrAddr()
              << " (" << std::dec << ddr_layout::BSR_PTR_SIZE << " bytes)\n";
    std::cout << std::hex;
    std::cout << "  BSR col_idx:0x" << std::setw(8) << bsrIdxAddr()
              << " (" << std::dec << ddr_layout::BSR_IDX_SIZE << " bytes)\n";
    std::cout << std::hex;
    std::cout << "  Output:     0x" << std::setw(8) << outputAddr()
              << " (" << std::dec << ddr_layout::OUTPUT_SIZE << " bytes)\n";

    std::cout << "\n-- Bank Selection --\n";
    std::cout << "  Weight Bank:     " << wgt_bank_ << "\n";
    std::cout << "  Activation Bank: " << act_bank_ << "\n";

    std::cout << "\n-- DMA Status --\n";
    uint32_t dma_stat = csr_.read32(REG_DMA_CTRL);
    uint32_t act_stat = csr_.read32(REG_ACT_DMA_CTRL);
    std::cout << "  Weight DMA:  "
              << ((dma_stat & dma_ctrl::BUSY_BIT) ? "BUSY" : "idle")
              << ((dma_stat & dma_ctrl::DONE_BIT) ? " DONE" : "") << "\n";
    std::cout << "  Act DMA:     "
              << ((act_stat & dma_ctrl::BUSY_BIT) ? "BUSY" : "idle")
              << ((act_stat & dma_ctrl::DONE_BIT) ? " DONE" : "") << "\n";

    std::cout << "\n-- Cumulative Statistics --\n";
    std::cout << "  Total Bytes Transferred: " << std::dec << total_bytes_ << "\n";
    std::cout << "  Transfer Count:          " << transfer_count_ << "\n";

    std::cout << "==============================\n\n";
}

} // namespace memory
} // namespace accel
