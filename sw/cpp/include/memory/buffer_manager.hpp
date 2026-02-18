// buffer_manager.hpp — Double-buffered ping-pong manager for zero-stall compute
// =============================================================================
//
// PURPOSE
// =======
// The hardware has double-buffered weight and activation SRAMs (wgt_buffer.sv,
// act_buffer.sv) and a double-buffered output accumulator (output_accumulator.sv).
// This class orchestrates the *software* side of the ping-pong:
//
//   While the systolic array computes from Bank A,
//   we DMA-prefetch the next tile's data into Bank B.
//
// When the tile is done, we swap: Bank A ↔ Bank B.  Zero idle cycles.
//
// OWNERSHIP
// =========
// BufferManager does NOT own DMAController or CSRInterface — it borrows them.
// Lifetime: BufferManager must not outlive the DMAController it references.
//
// RELATIONSHIP TO RTL
// ===================
//   act_buffer.sv   → bank_sel_wr / bank_sel_rd  (2 banks, TM=14 wide)
//   wgt_buffer.sv   → bank_sel_wr / bank_sel_rd  (2 banks, TN=14 wide)
//   output_accumulator.sv → bank_sel, tile_done   (2 banks, 196 accumulators)
//   csr.sv           → REG_BUFF[0]=act_wr, [1]=wgt_wr, [8]=act_rd, [9]=wgt_rd
//
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <string>
#include <stdexcept>
#include <functional>

// Forward declarations — avoid circular includes
namespace accel { namespace driver { class CSRInterface; } }
namespace accel { namespace memory { class DMAController;  } }

namespace accel {
namespace memory {

// =============================================================================
// Constants — mirror hw/rtl parameters and address_map.hpp
// =============================================================================
constexpr uint32_t NUM_BANKS       = 2;         // Ping-pong (0 and 1)
constexpr uint32_t TILE_DIM        = 14;        // Systolic array dimension
constexpr uint32_t TILE_ELEMENTS   = TILE_DIM * TILE_DIM;  // 196
constexpr uint32_t TILE_BYTES_INT8 = TILE_ELEMENTS;        // 196 bytes per tile
constexpr uint32_t ACC_WIDTH       = 32;        // Output accumulator bits

// =============================================================================
// BufferState — Tracks what each bank is doing right now
// =============================================================================
enum class BufferState : uint8_t {
    Idle,       // Bank is empty / available for loading
    Loading,    // DMA transfer in progress into this bank
    Ready,      // Data loaded, waiting for compute to consume it
    Computing,  // Systolic array is actively reading from this bank
    Draining    // Output DMA is reading results from this bank
};

const char* bufferStateName(BufferState s) noexcept;

// =============================================================================
// TileDescriptor — Identifies one (m, n, k) tile in the tiled GEMM
// =============================================================================
struct TileDescriptor {
    uint32_t m_idx;           // M-tile index  (row block)
    uint32_t n_idx;           // N-tile index  (column block)
    uint32_t k_idx;           // K-tile index  (reduction block)

    uint32_t wgt_offset;      // Byte offset into DDR weight region
    uint32_t wgt_bytes;       // Weight tile size in bytes
    uint32_t act_offset;      // Byte offset into DDR activation region
    uint32_t act_bytes;       // Activation tile size in bytes

    bool     is_first_k;      // First K-tile → clear accumulators
    bool     is_last_k;       // Last K-tile  → trigger output DMA
};

// =============================================================================
// BankInfo — Per-bank metadata (one per channel per bank)
// =============================================================================
struct BankInfo {
    BufferState state       = BufferState::Idle;
    TileDescriptor tile     = {};       // Which tile occupies this bank
    uint64_t load_start_us  = 0;        // Timestamp when DMA started
    uint64_t load_end_us    = 0;        // Timestamp when DMA completed
};

// =============================================================================
// BufferStats — Aggregate statistics for profiling / roofline analysis
// =============================================================================
struct BufferStats {
    // Tile-level counters
    uint32_t tiles_prefetched    = 0;   // Tiles loaded via DMA
    uint32_t tiles_computed      = 0;   // Tiles consumed by compute
    uint32_t tiles_drained       = 0;   // Output tiles written to DDR

    // Stall counters (the numbers that matter for roofline)
    uint32_t compute_stalls      = 0;   // Compute waited for DMA to finish
    uint32_t dma_stalls          = 0;   // DMA waited for bank to free up
    uint32_t output_stalls       = 0;   // Output drain blocked new accumulation

    // Byte counters
    uint64_t wgt_bytes_moved     = 0;   // Total weight bytes DMA'd
    uint64_t act_bytes_moved     = 0;   // Total activation bytes DMA'd
    uint64_t out_bytes_moved     = 0;   // Total output bytes DMA'd

    // Bank swap counter
    uint32_t bank_swaps          = 0;

    // Timing (microseconds)
    uint64_t total_time_us       = 0;
    uint64_t compute_time_us     = 0;
    uint64_t dma_time_us         = 0;
    uint64_t idle_time_us        = 0;

    // Derived metrics
    double   computeUtilisation() const;   // compute_time / total_time
    double   dmaOverlapRatio()   const;    // 1.0 = fully hidden
    double   weightBandwidthMBs() const;   // wgt_bytes / dma_time_us
    double   actBandwidthMBs()   const;
};

// =============================================================================
// BufferManager — The main orchestrator
// =============================================================================
//
// Typical call sequence (from Accelerator::runTiledGEMM):
//
//   BufferManager buf(dma, csr);
//   buf.reset();
//
//   // Prefetch first tile into bank 0
//   buf.prefetchWeights(tile_0);
//   buf.prefetchActivations(tile_0);
//   buf.waitPrefetchDone();
//
//   for (auto& tile : tiles) {
//       // Prefetch NEXT tile into the other bank (overlapped with compute)
//       if (has_next) {
//           buf.prefetchWeights(next_tile);
//           buf.prefetchActivations(next_tile);
//       }
//
//       // Compute current tile (systolic array reads from the ready bank)
//       buf.beginCompute(tile);
//       buf.waitComputeDone();
//
//       // If last K-tile, drain output
//       if (tile.is_last_k) {
//           buf.drainOutput();
//       }
//
//       // Swap banks
//       buf.swapBanks();
//       buf.waitPrefetchDone();   // Ensure next tile finished loading
//   }
//
class BufferManager {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// @param dma  Reference to an initialised DMAController (DDR must be mapped)
    /// @param csr  Reference to an initialised CSRInterface
    BufferManager(DMAController& dma, driver::CSRInterface& csr);
    ~BufferManager() = default;

    // Non-copyable (references external resources)
    BufferManager(const BufferManager&)            = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    // Movable
    BufferManager(BufferManager&&) noexcept;
    BufferManager& operator=(BufferManager&&) noexcept;

    // -------------------------------------------------------------------------
    // Initialisation / Reset
    // -------------------------------------------------------------------------

    /// Reset all bank states to Idle, zero statistics, set banks to 0.
    void reset();

    // -------------------------------------------------------------------------
    // Prefetch — DMA data into the inactive (write) bank
    // -------------------------------------------------------------------------

    /// Initiate weight DMA into the current write bank.
    /// @param tile  Tile descriptor with DDR offsets and byte counts.
    void prefetchWeights(const TileDescriptor& tile);

    /// Initiate activation DMA into the current write bank.
    void prefetchActivations(const TileDescriptor& tile);

    /// Convenience: prefetch both weights and activations for a tile.
    void prefetchTile(const TileDescriptor& tile);

    /// Block until the in-flight prefetch DMA(s) complete.
    /// @param timeout_us  Timeout in microseconds (0 = infinite).
    /// @return true if DMA completed, false on timeout.
    bool waitPrefetchDone(uint32_t timeout_us = 5000000);

    /// Check if prefetch is still in progress (non-blocking).
    bool isPrefetchBusy() const;

    // -------------------------------------------------------------------------
    // Compute — Mark a bank as "in use" by the systolic array
    // -------------------------------------------------------------------------

    /// Mark the current read bank as Computing and start the accelerator.
    /// The scheduler in bsr_scheduler.sv reads weights and activations from
    /// the read bank while DMA loads the next tile into the write bank.
    /// @param tile  The tile that was previously prefetched into this bank.
    void beginCompute(const TileDescriptor& tile);

    /// Block until the current tile's compute is done (polls STATUS[1] done_tile).
    /// @param timeout_us  Timeout in microseconds.
    /// @return true if compute completed, false on timeout.
    bool waitComputeDone(uint32_t timeout_us = 10000000);

    /// Check if compute is still running (non-blocking).
    bool isComputeBusy() const;

    // -------------------------------------------------------------------------
    // Output Drain — Read accumulated results from output buffer to DDR
    // -------------------------------------------------------------------------

    /// Configure output DMA to drain the inactive accumulator bank to DDR.
    /// Called after the last K-tile of an (m, n) output tile.
    void drainOutput();

    /// Block until output drain completes.
    bool waitDrainDone(uint32_t timeout_us = 5000000);

    // -------------------------------------------------------------------------
    // Bank Management
    // -------------------------------------------------------------------------

    /// Swap read/write banks for both weight and activation buffers.
    /// This is the core ping-pong operation.
    void swapBanks();

    /// Get the current write bank index (0 or 1)
    uint32_t writeBank() const { return write_bank_; }

    /// Get the current read (compute) bank index (0 or 1)
    uint32_t readBank()  const { return write_bank_ ^ 1; }

    // -------------------------------------------------------------------------
    // High-Level Tile Execution (convenience wrapper)
    // -------------------------------------------------------------------------

    /// Execute a full sequence of tiles with automatic double-buffering.
    /// @param tiles       Array of tile descriptors in execution order.
    /// @param num_tiles   Number of tiles.
    /// @param on_tile_done Optional callback after each tile completes.
    void executeTileSequence(const TileDescriptor* tiles, size_t num_tiles,
                             std::function<void(const TileDescriptor&, uint32_t)>
                                 on_tile_done = nullptr);

    // -------------------------------------------------------------------------
    // State Inspection
    // -------------------------------------------------------------------------

    /// Get the state of a specific weight bank
    BufferState weightBankState(uint32_t bank) const;

    /// Get the state of a specific activation bank
    BufferState actBankState(uint32_t bank) const;

    /// Get the state of a specific output bank
    BufferState outputBankState(uint32_t bank) const;

    /// Get cumulative statistics
    const BufferStats& stats() const { return stats_; }

    // -------------------------------------------------------------------------
    // Debug
    // -------------------------------------------------------------------------

    /// Dump full buffer state to stdout
    void dumpState() const;

private:
    // -------------------------------------------------------------------------
    // References to hardware interfaces (not owned)
    // -------------------------------------------------------------------------
    DMAController&        dma_;
    driver::CSRInterface& csr_;

    // -------------------------------------------------------------------------
    // Bank tracking (per-channel, per-bank)
    // -------------------------------------------------------------------------
    std::array<BankInfo, NUM_BANKS> wgt_banks_;
    std::array<BankInfo, NUM_BANKS> act_banks_;
    std::array<BankInfo, NUM_BANKS> out_banks_;

    // -------------------------------------------------------------------------
    // Current bank pointers
    // -------------------------------------------------------------------------
    uint32_t write_bank_;   // Bank that DMA writes into (0 or 1)
                            // Read bank = write_bank_ ^ 1

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------
    BufferStats stats_;

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Get current timestamp in microseconds (monotonic clock)
    static uint64_t nowUs();

    /// Programme the CSR bank-select bits for weight/activation write banks
    void applyBankSelect();

    /// Transition a bank to a new state with validation
    static void transition(BankInfo& bank, BufferState from, BufferState to);
};

// =============================================================================
// Exception
// =============================================================================
class BufferException : public std::runtime_error {
public:
    explicit BufferException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace memory
} // namespace accel
