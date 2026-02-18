// buffer_manager.cpp — Double-buffered ping-pong manager implementation
// =============================================================================
//
// Orchestrates weight, activation, and output double-buffering so that DMA
// transfers overlap with systolic array compute.  Zero-stall when DMA latency
// is hidden behind compute-bound tiles.
//
// Relationship to hardware:
//   csr.sv REG_BUFF[0] = act write bank, [1] = wgt write bank
//   Hardware read bank is always the opposite of write bank (XOR with 1).
//   output_accumulator.sv manages its own bank_sel via tile_done pulse.
//
// =============================================================================
#include "memory/buffer_manager.hpp"
#include "memory/dma_controller.hpp"
#include "memory/address_map.hpp"
#include "driver/csr_interface.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cassert>

namespace accel {
namespace memory {

using namespace accel::memory;  // address_map constants

// =============================================================================
// BufferState name helper
// =============================================================================

const char* bufferStateName(BufferState s) noexcept {
    switch (s) {
        case BufferState::Idle:      return "Idle";
        case BufferState::Loading:   return "Loading";
        case BufferState::Ready:     return "Ready";
        case BufferState::Computing: return "Computing";
        case BufferState::Draining:  return "Draining";
    }
    return "Unknown";
}

// =============================================================================
// BufferStats derived metrics
// =============================================================================

double BufferStats::computeUtilisation() const {
    return (total_time_us > 0)
               ? static_cast<double>(compute_time_us) / total_time_us
               : 0.0;
}

double BufferStats::dmaOverlapRatio() const {
    // Ideal: DMA time is fully hidden behind compute → ratio = 1.0
    // Worst: DMA time is all exposed (stalls) → ratio = 0.0
    if (dma_time_us == 0) return 1.0;
    uint64_t exposed = (compute_stalls > 0)
        ? static_cast<uint64_t>(compute_stalls)  // rough proxy
        : 0;
    double hidden = static_cast<double>(dma_time_us - exposed);
    return (hidden > 0.0) ? hidden / dma_time_us : 0.0;
}

double BufferStats::weightBandwidthMBs() const {
    return (dma_time_us > 0)
               ? static_cast<double>(wgt_bytes_moved) / dma_time_us  // bytes/µs = MB/s
               : 0.0;
}

double BufferStats::actBandwidthMBs() const {
    return (dma_time_us > 0)
               ? static_cast<double>(act_bytes_moved) / dma_time_us
               : 0.0;
}

// =============================================================================
// Timestamp helper (monotonic, microsecond resolution)
// =============================================================================

uint64_t BufferManager::nowUs() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count());
}

// =============================================================================
// Constructor
// =============================================================================

BufferManager::BufferManager(DMAController& dma, driver::CSRInterface& csr)
    : dma_(dma)
    , csr_(csr)
    , wgt_banks_()
    , act_banks_()
    , out_banks_()
    , write_bank_(0)
    , stats_()
{
    // Ensure bank arrays are zero-initialised (BankInfo defaults apply)
}

// =============================================================================
// Move construction / assignment
// =============================================================================

BufferManager::BufferManager(BufferManager&& other) noexcept
    : dma_(other.dma_)
    , csr_(other.csr_)
    , wgt_banks_(other.wgt_banks_)
    , act_banks_(other.act_banks_)
    , out_banks_(other.out_banks_)
    , write_bank_(other.write_bank_)
    , stats_(other.stats_)
{
}

BufferManager& BufferManager::operator=(BufferManager&& other) noexcept {
    if (this != &other) {
        // csr_ and dma_ are references — must already alias the same object
        wgt_banks_  = other.wgt_banks_;
        act_banks_  = other.act_banks_;
        out_banks_  = other.out_banks_;
        write_bank_ = other.write_bank_;
        stats_      = other.stats_;
    }
    return *this;
}

// =============================================================================
// State transition with validation
// =============================================================================

void BufferManager::transition(BankInfo& bank, BufferState from, BufferState to) {
    if (bank.state != from) {
        throw BufferException(
            std::string("Invalid buffer state transition: expected ") +
            bufferStateName(from) + " but current state is " +
            bufferStateName(bank.state));
    }
    bank.state = to;
}

// =============================================================================
// Reset
// =============================================================================

void BufferManager::reset() {
    write_bank_ = 0;

    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        wgt_banks_[b] = BankInfo{};
        act_banks_[b] = BankInfo{};
        out_banks_[b] = BankInfo{};
    }

    stats_ = BufferStats{};

    // Set hardware bank select to 0
    applyBankSelect();
}

// =============================================================================
// Bank Select — Programme CSR REG_BUFF
// =============================================================================
//
// REG_BUFF layout (from csr.sv / address_map.hpp):
//   [0] = act write bank select   (RW)
//   [1] = wgt write bank select   (RW)
//   [8] = act read bank           (RO — hardware derives as ~write)
//   [9] = wgt read bank           (RO — hardware derives as ~write)
//
void BufferManager::applyBankSelect() {
    uint32_t buff_val = csr_.read32(REG_BUFF);

    // Clear bits [1:0] and set to current write_bank_
    buff_val &= ~0x3u;
    if (write_bank_ & 1u) {
        buff_val |= 0x3u;   // Both act and wgt write to bank 1
    }
    // else: both stay 0  (bank 0)

    csr_.write32(REG_BUFF, buff_val);

    // Also sync the DMAController's shadow state so dumpState() is consistent
    dma_.setWeightBank(write_bank_);
    dma_.setActivationBank(write_bank_);
}

// =============================================================================
// Prefetch — DMA data into the write bank
// =============================================================================

void BufferManager::prefetchWeights(const TileDescriptor& tile) {
    BankInfo& bank = wgt_banks_[write_bank_];

    // The bank must be Idle (or Ready from a previous unused prefetch)
    if (bank.state != BufferState::Idle && bank.state != BufferState::Ready) {
        stats_.dma_stalls++;
        // Wait for the bank to become free (it might be draining)
        // This is the "DMA stall" case — DMA has nowhere to write
    }

    bank.tile          = tile;
    bank.load_start_us = nowUs();
    bank.state         = BufferState::Loading;

    // Initiate DMA: copy tile.wgt_bytes from DDR offset tile.wgt_offset
    // into the hardware weight buffer, write bank = write_bank_.
    //
    // The DMA path:  DDR → AXI → axi_dma_bridge → bsr_dma → wgt_buffer
    //
    // dma_.loadWeights() puts data in DDR (host → DDR via mmap).
    // dma_.startWeightTransfer() tells the hardware DMA to pull it.
    //
    // BUT: if the host has already staged the full weight matrix at init time,
    // we only need to tell the hardware DMA the source address and length.
    dma_.startWeightTransfer(tile.wgt_bytes);

    stats_.wgt_bytes_moved += tile.wgt_bytes;
    stats_.tiles_prefetched++;
}

void BufferManager::prefetchActivations(const TileDescriptor& tile) {
    BankInfo& bank = act_banks_[write_bank_];

    if (bank.state != BufferState::Idle && bank.state != BufferState::Ready) {
        stats_.dma_stalls++;
    }

    bank.tile          = tile;
    bank.load_start_us = nowUs();
    bank.state         = BufferState::Loading;

    dma_.startActivationTransfer(tile.act_bytes);

    stats_.act_bytes_moved += tile.act_bytes;
}

void BufferManager::prefetchTile(const TileDescriptor& tile) {
    prefetchWeights(tile);
    prefetchActivations(tile);
}

// =============================================================================
// Wait for prefetch DMA to complete
// =============================================================================

bool BufferManager::waitPrefetchDone(uint32_t timeout_us) {
    bool wgt_ok = dma_.waitWeightDone(timeout_us);
    bool act_ok = dma_.waitActivationDone(timeout_us);

    uint64_t now = nowUs();

    // Mark weight bank as Ready
    BankInfo& wb = wgt_banks_[write_bank_];
    if (wgt_ok && wb.state == BufferState::Loading) {
        wb.state       = BufferState::Ready;
        wb.load_end_us = now;
        stats_.dma_time_us += (wb.load_end_us - wb.load_start_us);
    }

    // Mark activation bank as Ready
    BankInfo& ab = act_banks_[write_bank_];
    if (act_ok && ab.state == BufferState::Loading) {
        ab.state       = BufferState::Ready;
        ab.load_end_us = now;
        stats_.dma_time_us += (ab.load_end_us - ab.load_start_us);
    }

    return wgt_ok && act_ok;
}

bool BufferManager::isPrefetchBusy() const {
    return dma_.isAnyBusy();
}

// =============================================================================
// Compute — Drive the systolic array from the read bank
// =============================================================================

void BufferManager::beginCompute(const TileDescriptor& tile) {
    uint32_t rb = readBank();  // The bank the systolic array reads from

    BankInfo& wb = wgt_banks_[rb];
    BankInfo& ab = act_banks_[rb];

    // The read bank should have been loaded and marked Ready
    if (wb.state != BufferState::Ready) {
        stats_.compute_stalls++;
        throw BufferException("Weight bank " + std::to_string(rb) +
                              " is not Ready for compute (state=" +
                              bufferStateName(wb.state) + ")");
    }
    if (ab.state != BufferState::Ready) {
        stats_.compute_stalls++;
        throw BufferException("Activation bank " + std::to_string(rb) +
                              " is not Ready for compute");
    }

    wb.state = BufferState::Computing;
    ab.state = BufferState::Computing;

    // If first K-tile, clear accumulators
    if (tile.is_first_k) {
        // acc_clear is pulsed via status register or CSR write.  The scheduler
        // handles this internally when we write new tile indices and start.
    }

    // Programme tile dimensions and indices into CSR
    csr_.write32(REG_INDEX_M, tile.m_idx);
    csr_.write32(REG_INDEX_N, tile.n_idx);
    csr_.write32(REG_INDEX_K, tile.k_idx);

    // Pulse start (W1P on CTRL[0])
    csr_.start();
}

bool BufferManager::waitComputeDone(uint32_t timeout_us) {
    uint64_t t0 = nowUs();

    bool ok = csr_.waitDone(timeout_us);

    uint64_t elapsed = nowUs() - t0;
    stats_.compute_time_us += elapsed;
    stats_.tiles_computed++;

    // Transition read-bank states back to Idle
    uint32_t rb = readBank();
    wgt_banks_[rb].state = BufferState::Idle;
    act_banks_[rb].state = BufferState::Idle;

    // Clear done_tile bit (W1C on STATUS[1])
    csr_.clearBits(REG_STATUS, status::DONE_TILE_BIT);

    return ok;
}

bool BufferManager::isComputeBusy() const {
    return csr_.isBusy();
}

// =============================================================================
// Output Drain
// =============================================================================

void BufferManager::drainOutput() {
    // The output accumulator bank that just finished computing is the
    // *hardware* output bank — it auto-swaps on tile_done.
    // We configure output DMA to pull from the inactive accumulator bank.

    dma_.configureOutputDMA();

    // Track output bank state
    uint32_t ob = readBank();  // The bank that was just computing
    out_banks_[ob].state = BufferState::Draining;
}

bool BufferManager::waitDrainDone(uint32_t timeout_us) {
    // Output DMA auto-triggers and writes to DDR.  We poll the output DMA
    // status via the CSR.
    // For now we poll the generic status — out_dma.sv raises its own done flag.
    // We use a simple busy-wait approach consistent with DMAController.

    uint64_t deadline = nowUs() + timeout_us;

    while (nowUs() < deadline) {
        // Check if output area has been written (or DMA done bit is set)
        uint32_t stat = csr_.read32(REG_STATUS);
        if (!(stat & status::BUSY_BIT)) {
            // Mark output bank idle
            for (uint32_t b = 0; b < NUM_BANKS; ++b) {
                if (out_banks_[b].state == BufferState::Draining) {
                    out_banks_[b].state = BufferState::Idle;
                }
            }
            stats_.tiles_drained++;
            stats_.out_bytes_moved += TILE_BYTES_INT8;
            return true;
        }
    }

    stats_.output_stalls++;
    return false;
}

// =============================================================================
// Bank Swap — The core ping-pong operation
// =============================================================================

void BufferManager::swapBanks() {
    write_bank_ ^= 1u;     // Toggle 0 ↔ 1
    applyBankSelect();      // Update CSR REG_BUFF and DMAController
    stats_.bank_swaps++;
}

// =============================================================================
// State Inspection
// =============================================================================

BufferState BufferManager::weightBankState(uint32_t bank) const {
    assert(bank < NUM_BANKS);
    return wgt_banks_[bank].state;
}

BufferState BufferManager::actBankState(uint32_t bank) const {
    assert(bank < NUM_BANKS);
    return act_banks_[bank].state;
}

BufferState BufferManager::outputBankState(uint32_t bank) const {
    assert(bank < NUM_BANKS);
    return out_banks_[bank].state;
}

// =============================================================================
// High-Level Tile Execution (double-buffered loop)
// =============================================================================
//
//  Timeline (ideal — DMA fully hidden):
//
//    Bank 0:  [DMA tile0] [COMPUTE tile0]             [DMA tile2] [COMPUTE tile2]
//    Bank 1:              [DMA tile1]    [COMPUTE tile1]           [DMA tile3] ...
//
//  If DMA is slower than compute, we stall (compute_stalls++).
//  If compute is slower than DMA, the DMA finishes early (no stall).
//

void BufferManager::executeTileSequence(
    const TileDescriptor* tiles,
    size_t num_tiles,
    std::function<void(const TileDescriptor&, uint32_t)> on_tile_done)
{
    if (num_tiles == 0) return;

    uint64_t seq_start = nowUs();
    reset();

    // ── Phase 1: Prefetch first tile into bank 0 ────────────────────────────
    prefetchTile(tiles[0]);
    if (!waitPrefetchDone()) {
        throw BufferException("Initial prefetch timed out");
    }

    // ── Phase 2: Double-buffered main loop ──────────────────────────────────
    for (size_t i = 0; i < num_tiles; ++i) {
        const TileDescriptor& current = tiles[i];
        bool has_next = (i + 1 < num_tiles);

        // ── Prefetch next tile into write bank (overlapped with compute) ────
        if (has_next) {
            prefetchTile(tiles[i + 1]);
        }

        // ── Compute current tile from read bank ────────────────────────────
        beginCompute(current);

        if (!waitComputeDone()) {
            throw BufferException("Compute timed out on tile " + std::to_string(i));
        }

        // ── Drain output if this is the last K-tile ────────────────────────
        if (current.is_last_k) {
            drainOutput();
            if (!waitDrainDone()) {
                throw BufferException("Output drain timed out on tile " +
                                      std::to_string(i));
            }
        }

        // ── User callback ──────────────────────────────────────────────────
        if (on_tile_done) {
            on_tile_done(current, static_cast<uint32_t>(i));
        }

        // ── Swap banks ─────────────────────────────────────────────────────
        if (has_next) {
            swapBanks();

            // Ensure next tile's DMA has finished before we try to compute it
            if (!waitPrefetchDone()) {
                stats_.compute_stalls++;
                throw BufferException("Prefetch stall on tile " +
                                      std::to_string(i + 1));
            }
        }
    }

    uint64_t seq_end = nowUs();
    stats_.total_time_us = seq_end - seq_start;
    stats_.idle_time_us  = stats_.total_time_us
                           - stats_.compute_time_us
                           - stats_.dma_time_us;
}

// =============================================================================
// Debug Output
// =============================================================================

void BufferManager::dumpState() const {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout <<   "║         BUFFER MANAGER STATE                 ║\n";
    std::cout <<   "╠══════════════════════════════════════════════╣\n";

    std::cout << "║ Write Bank:  " << write_bank_
              << "   Read Bank:  " << readBank() << "\n";

    auto printBank = [](const char* name, const std::array<BankInfo, NUM_BANKS>& banks) {
        for (uint32_t b = 0; b < NUM_BANKS; ++b) {
            std::cout << "║   " << name << "[" << b << "]: "
                      << bufferStateName(banks[b].state);
            if (banks[b].state != BufferState::Idle) {
                std::cout << "  (m=" << banks[b].tile.m_idx
                          << " n=" << banks[b].tile.n_idx
                          << " k=" << banks[b].tile.k_idx << ")";
            }
            std::cout << "\n";
        }
    };

    std::cout << "║\n║ ── Weight Banks ──\n";
    printBank("WGT", wgt_banks_);

    std::cout << "║\n║ ── Activation Banks ──\n";
    printBank("ACT", act_banks_);

    std::cout << "║\n║ ── Output Banks ──\n";
    printBank("OUT", out_banks_);

    std::cout << "║\n║ ── Statistics ──\n";
    std::cout << "║   Tiles prefetched:  " << stats_.tiles_prefetched  << "\n";
    std::cout << "║   Tiles computed:    " << stats_.tiles_computed    << "\n";
    std::cout << "║   Tiles drained:     " << stats_.tiles_drained    << "\n";
    std::cout << "║   Bank swaps:        " << stats_.bank_swaps       << "\n";
    std::cout << "║   Compute stalls:    " << stats_.compute_stalls   << "\n";
    std::cout << "║   DMA stalls:        " << stats_.dma_stalls       << "\n";
    std::cout << "║   Output stalls:     " << stats_.output_stalls    << "\n";
    std::cout << "║\n";
    std::cout << "║   Weight bytes:      " << stats_.wgt_bytes_moved  << "\n";
    std::cout << "║   Activation bytes:  " << stats_.act_bytes_moved  << "\n";
    std::cout << "║   Output bytes:      " << stats_.out_bytes_moved  << "\n";
    std::cout << "║\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "║   Total time:        " << stats_.total_time_us   << " µs\n";
    std::cout << "║   Compute time:      " << stats_.compute_time_us << " µs\n";
    std::cout << "║   DMA time:          " << stats_.dma_time_us     << " µs\n";
    std::cout << "║   Idle time:         " << stats_.idle_time_us    << " µs\n";
    std::cout << "║   Compute util:      " << stats_.computeUtilisation() * 100.0 << "%\n";
    std::cout << "║   DMA overlap:       " << stats_.dmaOverlapRatio() * 100.0 << "%\n";
    std::cout << "║   Wgt BW:            " << stats_.weightBandwidthMBs() << " MB/s\n";
    std::cout << "║   Act BW:            " << stats_.actBandwidthMBs()   << " MB/s\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";
}

} // namespace memory
} // namespace accel
