# RTL Memory Improvements & C++ Implementation Guide

## For RA Position - Memory Architecture Focus

---

## Part 1: Current Memory Architecture Analysis

### What You Have (14×14 Systolic Array on Pynq-Z2)

| Component | Current Implementation | Memory Type |
|-----------|----------------------|-------------|
| Activation Buffer | Inline BRAM (1024×112b) | ~14 KB |
| Weight Block BRAM | Inline BRAM (1024×112b) | ~14 KB |
| Row Pointer BRAM | Inline BRAM (1024×32b) | ~4 KB |
| Column Index BRAM | Inline BRAM (1024×16b) | ~2 KB |
| Output Accumulator | Double-buffered (2×196×32b) | ~1.6 KB |
| **Total On-Chip** | | **~36 KB** |

### Current Data Flow

```
DDR → AXI HP (64-bit) → act_dma/bsr_dma → dma_pack_112 → BRAM → Systolic → Output Accum → DDR
                        (2:1 arbiter)      (64→112 packer)
```

### Memory-Bound Analysis

**Compute Capacity:**
- 196 PEs × 1 MAC/cycle × 100 MHz = **19.6 GOPS**

**Bandwidth Required (to sustain full utilization):**
- 196 MACs × 2 bytes (weight + activation) × 100 MHz = **39.2 GB/s**

**Zynq HP Port Provides:**
- ~2-4 GB/s practical

**Conclusion:** You're **~10-20× memory-bound** without on-chip buffering strategies.

---

## Part 2: RTL Improvements for Better Memory Usage

### 2.1 ALREADY IMPLEMENTED (Good Foundations)

| Feature | What It Does | Why It Helps |
|---------|-------------|--------------|
| **BSR Sparsity** | Skips zero blocks entirely | Reduces effective bandwidth by sparsity ratio |
| **Weight-Stationary Dataflow** | Weights loaded once, activations stream | 14× weight reuse per block |
| **64→112 Packer** | Converts AXI beats to BRAM width | No bandwidth waste on width mismatch |
| **Output Accumulator Double-Buffer** | Ping-pong banks for compute/DMA overlap | Zero stall between tiles |
| **Clock Gating** | Gate buffers when idle | Power savings |

### 2.2 CAN ADD (Medium Effort, 1-2 Weeks RTL)

#### A. Activation Buffer Double-Buffering

**Current Problem:** 
- `act_buffer_ram` is single-buffered in `accel_top.sv`
- DMA must complete before compute starts

**Solution:**
```systemverilog
// Add in accel_top.sv - true double-buffering
reg [N_ROWS*DATA_W-1:0] act_buffer_ram_bank0 [0:(1<<BRAM_ADDR_W)-1];
reg [N_ROWS*DATA_W-1:0] act_buffer_ram_bank1 [0:(1<<BRAM_ADDR_W)-1];
reg act_bank_sel;

// DMA writes to bank[~act_bank_sel] while compute reads from bank[act_bank_sel]
```

**Benefit:** Overlap activation DMA with previous tile computation.

**BRAM Cost:** +14 KB (doubles act buffer)

#### B. Weight Block Streaming (Reduce BRAM)

**Current:** All BSR weight blocks loaded to BRAM before compute starts.

**Alternative:** Stream weight blocks on-demand from DDR.

```systemverilog
// Add prefetch state to bsr_scheduler.sv:
// While computing block[n], prefetch block[n+1] into shadow buffer
```

**Benefit:** Reduces wgt_block_bram to 2× block size (~400 bytes instead of 14 KB).

**Risk:** Requires careful timing; DMA latency must be hidden.

#### C. Performance Counter Enhancements

**Add to `perf.sv`:**
```systemverilog
// Bandwidth utilization counter
reg [31:0] dma_stall_cycles;  // Cycles DMA wanted to send but couldn't
reg [31:0] bram_conflict_cycles;  // Read/write port conflicts

// Roofline model support
wire [31:0] arithmetic_intensity = perf_active_cycles / perf_dma_bytes;
```

**Why:** Proves to reviewers you understand memory architecture metrics.

### 2.3 OPTIONAL (Higher Effort, 2-3 Weeks)

#### D. AXI Burst Optimization

**Current:** Fixed 16-beat bursts (128 bytes)

**Improvement:** Adaptive burst length based on transfer size.

```systemverilog
// In act_dma.sv / bsr_dma.sv:
wire [7:0] optimal_burst_len = (remaining_bytes >= 128) ? 8'd15 :
                               (remaining_bytes >= 64)  ? 8'd7  :
                               (remaining_bytes >= 32)  ? 8'd3  : 8'd0;
```

#### E. Read-Modify-Write for Partial Tiles

For matrices not divisible by 14, edge tiles waste compute.

**Solution:** Mask-based accumulation for partial tiles.

---

## Part 3: C++ Implementation Details

### 3.1 File-by-File Implementation Guide

#### `include/memory/address_map.hpp`

**Purpose:** Hardware register definitions - THE most critical file for memory architecture.

```cpp
namespace accel::memory {

// Pynq-Z2 PL address space
constexpr uint32_t ACCEL_BASE = 0x43C00000;

// CSR Register Map (from csr.sv)
struct CSRMap {
    static constexpr uint32_t CTRL         = 0x00;  // [2]=irq_en, [1]=abort, [0]=start
    static constexpr uint32_t DIMS_M       = 0x04;
    static constexpr uint32_t DIMS_N       = 0x08;
    static constexpr uint32_t DIMS_K       = 0x0C;
    // ... (match all addresses from csr.sv)
    
    // DMA Registers
    static constexpr uint32_t DMA_SRC_ADDR = 0x90;  // BSR base in DDR
    static constexpr uint32_t ACT_DMA_SRC  = 0xA0;  // Activation base in DDR
    static constexpr uint32_t ACT_DMA_LEN  = 0xA4;  // Bytes to transfer
    static constexpr uint32_t ACT_DMA_CTRL = 0xA8;  // [0]=start
    
    // BSR Configuration
    static constexpr uint32_t BSR_CONFIG     = 0xC0;
    static constexpr uint32_t BSR_NUM_BLOCKS = 0xC4;
    static constexpr uint32_t BSR_BLOCK_ROWS = 0xC8;
    static constexpr uint32_t BSR_BLOCK_COLS = 0xCC;
    
    // Performance Counters (READ THESE FOR YOUR THESIS)
    static constexpr uint32_t PERF_TOTAL   = 0x40;
    static constexpr uint32_t PERF_ACTIVE  = 0x44;
    static constexpr uint32_t PERF_IDLE    = 0x48;
    static constexpr uint32_t PERF_DMA_BYTES    = 0x4C;
    static constexpr uint32_t PERF_BLOCKS_DONE  = 0x50;
    static constexpr uint32_t PERF_STALL_CYCLES = 0x54;
};

// DDR Memory Layout
struct DDRLayout {
    static constexpr uint32_t ACTIVATION_BASE = 0x10000000;  // 256 MB offset
    static constexpr uint32_t WEIGHT_BASE     = 0x10100000;  // +1 MB
    static constexpr uint32_t OUTPUT_BASE     = 0x10200000;  // +2 MB
};

}
```

#### `include/memory/dma_controller.hpp`

**Purpose:** Encapsulate DMA transfer logic.

```cpp
namespace accel::memory {

class DMAController {
public:
    // Memory-mapped I/O pointer (from mmap)
    explicit DMAController(volatile uint32_t* csr_base);
    
    // Start activation transfer
    void start_activation_dma(uint32_t ddr_addr, uint32_t length_bytes);
    
    // Start BSR weight transfer
    void start_bsr_dma(uint32_t ddr_addr, uint32_t num_blocks, 
                       uint32_t block_rows, uint32_t block_cols);
    
    // Poll for completion (blocking)
    void wait_dma_complete();
    
    // Non-blocking check
    bool is_busy() const;
    
    // Get bytes transferred (for bandwidth calculation)
    uint32_t get_bytes_transferred() const;
    
private:
    volatile uint32_t* csr_;
    
    // Helper: write to CSR
    void write_csr(uint32_t offset, uint32_t value);
    uint32_t read_csr(uint32_t offset) const;
};

}
```

#### `include/memory/buffer_manager.hpp`

**Purpose:** Manage double-buffering strategy (THIS IS KEY FOR MEMORY ARCHITECTURE).

```cpp
namespace accel::memory {

// Double-buffer state for zero-stall pipeline
class BufferManager {
public:
    enum class Bank { A = 0, B = 1 };
    
    BufferManager(DMAController& dma);
    
    // Prefetch next tile's data into inactive bank
    void prefetch_tile(uint32_t tile_idx, const TileDescriptor& desc);
    
    // Swap banks (after computation completes)
    void swap_banks();
    
    // Get currently active bank for compute
    Bank get_compute_bank() const { return compute_bank_; }
    
    // CRITICAL: Check if prefetch is complete before swap
    bool prefetch_ready() const;
    
    // Memory utilization stats
    struct Stats {
        uint32_t prefetch_stalls;  // Times compute waited for prefetch
        uint32_t successful_overlaps;  // Times prefetch completed in time
    };
    Stats get_stats() const;
    
private:
    DMAController& dma_;
    Bank compute_bank_ = Bank::A;
    bool prefetch_in_progress_ = false;
};

}
```

#### `include/driver/performance.hpp`

**Purpose:** THE FILE THAT PROVES YOU UNDERSTAND MEMORY ARCHITECTURE.

```cpp
namespace accel::driver {

// Maps directly to RTL perf.sv counters
struct PerformanceCounters {
    uint32_t total_cycles;
    uint32_t active_cycles;
    uint32_t idle_cycles;
    uint32_t dma_bytes;
    uint32_t blocks_processed;
    uint32_t stall_cycles;
};

class PerformanceAnalyzer {
public:
    explicit PerformanceAnalyzer(volatile uint32_t* csr_base);
    
    // Read counters from hardware
    PerformanceCounters read_counters() const;
    
    // Derived metrics (THIS IS WHAT INTERVIEWERS WANT TO SEE)
    struct Metrics {
        double compute_utilization;     // active_cycles / total_cycles
        double memory_utilization;      // dma_bytes / (total_cycles * peak_bw)
        double bandwidth_gbps;          // dma_bytes / total_time
        double gops;                    // (M*K*2*blocks) / total_cycles * freq
        double arithmetic_intensity;    // ops / bytes (roofline model)
        bool is_compute_bound;          // arithmetic_intensity > ridge_point
    };
    
    Metrics compute_metrics(const PerformanceCounters& counters,
                           uint32_t clock_freq_mhz = 100) const;
    
    // Print roofline analysis
    void print_roofline_analysis(const Metrics& m) const;
    
private:
    volatile uint32_t* csr_;
    
    // Pynq-Z2 theoretical peak bandwidth (GB/s)
    static constexpr double PEAK_BANDWIDTH = 4.0;
    
    // Ridge point for roofline (ops/byte)
    static constexpr double RIDGE_POINT = 19.6 / 4.0;  // peak_gops / peak_bw
};

}
```

#### `include/compute/tiling.hpp`

**Purpose:** Tile strategy for MNIST layers.

```cpp
namespace accel::compute {

// Systolic array is 14×14
constexpr int TILE_SIZE = 14;

// MNIST layer dimensions
struct LayerDims {
    int M;  // Output rows (batch * output_features)
    int N;  // Output cols (always 1 for FC, or output_features)
    int K;  // Inner dimension (input_features)
};

// Pre-computed tiling for MNIST FC layers
// Layer 1: 784 → 128 (56 K-tiles, 10 M-tiles)
// Layer 2: 128 → 10 (10 K-tiles, 1 M-tile)
struct TilingStrategy {
    int M_tiles;
    int K_tiles;
    int N_tiles;
    int total_tiles;
    
    static TilingStrategy compute(const LayerDims& dims);
};

// BSR block layout for a weight matrix
struct BSRLayout {
    int block_rows;       // M / TILE_SIZE
    int block_cols;       // K / TILE_SIZE
    int nnz_blocks;       // Non-zero blocks after pruning
    float sparsity;       // 1 - (nnz_blocks / total_blocks)
    
    // Memory calculation (for thesis)
    size_t row_ptr_bytes() const { return (block_rows + 1) * 4; }
    size_t col_idx_bytes() const { return nnz_blocks * 2; }
    size_t block_data_bytes() const { return nnz_blocks * TILE_SIZE * TILE_SIZE; }
    size_t total_bytes() const;
    
    // Compare to dense
    size_t dense_bytes() const { return block_rows * block_cols * TILE_SIZE * TILE_SIZE; }
    float compression_ratio() const { return (float)dense_bytes() / total_bytes(); }
};

}
```

#### `include/compute/bsr_encoder.hpp`

**Purpose:** Convert dense weights to BSR format for hardware.

```cpp
namespace accel::compute {

// BSR format header (matches RTL bsr_dma.sv expectations)
struct BSRHeader {
    uint32_t num_rows;      // Block rows
    uint32_t num_cols;      // Block cols
    uint32_t nnz_blocks;    // Non-zero blocks
    uint32_t block_size;    // Always 14
};

class BSREncoder {
public:
    // Prune threshold: blocks with all values < threshold become zero
    explicit BSREncoder(float prune_threshold = 0.0f);
    
    // Encode dense INT8 matrix to BSR
    // Input: row-major dense matrix [M × K]
    // Output: BSR structure ready for DMA
    struct BSRData {
        BSRHeader header;
        std::vector<uint32_t> row_ptr;
        std::vector<uint16_t> col_idx;
        std::vector<int8_t> blocks;  // Flattened block data
        
        // Get memory layout for DDR placement
        size_t total_bytes() const;
        void serialize_to(uint8_t* ddr_ptr) const;
    };
    
    BSRData encode(const int8_t* dense, int M, int K);
    
    // Analysis: sparsity stats for thesis
    struct SparsityStats {
        int total_blocks;
        int nonzero_blocks;
        int zero_blocks;
        float block_sparsity;
        float element_sparsity;  // If tracking individual zeros
    };
    SparsityStats analyze(const int8_t* dense, int M, int K);
    
private:
    float threshold_;
    
    bool is_block_nonzero(const int8_t* block_start, int M, int K, 
                          int block_row, int block_col);
};

}
```

### 3.2 Implementation Priority for 6 Weeks

| Week | Files to Implement | Why This Order |
|------|-------------------|----------------|
| 1 | `address_map.hpp`, `csr_interface.cpp` | Must talk to hardware first |
| 2 | `dma_controller.cpp`, basic transfers | DMA is core memory interface |
| 3 | `npy_loader.cpp`, `bsr_encoder.cpp` | Load test data, encode weights |
| 4 | `tiling.cpp`, `accelerator.cpp` | Full inference pipeline |
| 5 | `performance.cpp`, `benchmark.cpp` | **CRITICAL for RA position** |
| 6 | Polish, `run_mnist_inference.cpp` | End-to-end demo |

### 3.3 Key Code Patterns for Memory Architecture

#### Pattern 1: Memory-Mapped I/O (Essential)

```cpp
// In csr_interface.cpp
#include <fcntl.h>
#include <sys/mman.h>

class CSRInterface {
public:
    CSRInterface(uint32_t base_addr, size_t size = 4096) {
        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        csr_ = (volatile uint32_t*)mmap(nullptr, size,
                                        PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, base_addr);
        close(fd);
    }
    
    void write32(uint32_t offset, uint32_t value) {
        csr_[offset >> 2] = value;
        __sync_synchronize();  // Memory barrier
    }
    
    uint32_t read32(uint32_t offset) const {
        __sync_synchronize();
        return csr_[offset >> 2];
    }
};
```

#### Pattern 2: DMA Transfer with Barrier

```cpp
// In dma_controller.cpp
void DMAController::start_activation_dma(uint32_t addr, uint32_t len) {
    // Set source address
    write_csr(CSRMap::ACT_DMA_SRC, addr);
    write_csr(CSRMap::ACT_DMA_LEN, len);
    
    // Memory barrier before triggering
    __sync_synchronize();
    
    // Start (write-1-pulse)
    write_csr(CSRMap::ACT_DMA_CTRL, 0x1);
}

void DMAController::wait_dma_complete() {
    // Poll busy bit with exponential backoff
    int backoff = 1;
    while (read_csr(CSRMap::STATUS) & 0x1) {  // busy bit
        usleep(backoff);
        backoff = std::min(backoff * 2, 1000);  // Cap at 1ms
    }
}
```

#### Pattern 3: Performance Measurement

```cpp
// In benchmark.cpp - THIS IS WHAT GETS YOU THE RA POSITION
void run_benchmark() {
    PerformanceAnalyzer perf(csr_base);
    
    // Reset counters
    accelerator.reset();
    
    // Run inference
    auto start = std::chrono::high_resolution_clock::now();
    accelerator.run_inference(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Read hardware counters
    auto counters = perf.read_counters();
    auto metrics = perf.compute_metrics(counters);
    
    // Print analysis
    printf("=== Memory Architecture Analysis ===\n");
    printf("Compute Utilization: %.1f%%\n", metrics.compute_utilization * 100);
    printf("Memory Bandwidth: %.2f GB/s\n", metrics.bandwidth_gbps);
    printf("Arithmetic Intensity: %.2f ops/byte\n", metrics.arithmetic_intensity);
    printf("Bound: %s\n", metrics.is_compute_bound ? "COMPUTE" : "MEMORY");
    
    // Roofline
    printf("\n=== Roofline Model ===\n");
    printf("Peak Compute: 19.6 GOPS\n");
    printf("Peak Bandwidth: 4.0 GB/s\n");
    printf("Ridge Point: 4.9 ops/byte\n");
    printf("Your AI: %.2f (%.1f%% of ridge)\n", 
           metrics.arithmetic_intensity,
           metrics.arithmetic_intensity / 4.9 * 100);
}
```

---

## Part 4: What to Say in Interviews

### "Explain your memory architecture"

> "I implemented a weight-stationary systolic array with BSR sparsity. The key 
> insight is that on a resource-constrained FPGA like Pynq-Z2, you're inherently 
> memory-bound - my 196-PE array can do 19.6 GOPS but the HP port only delivers 
> ~4 GB/s. I addressed this through:
> 
> 1. **BSR compression** - skipping zero blocks reduces effective bandwidth demand
> 2. **Weight-stationary dataflow** - each weight is reused 14× per block
> 3. **Double-buffered output accumulator** - overlaps compute with DMA
> 4. **64→112 bit packing** - eliminates width mismatch waste
>
> My performance counters show [X]% compute utilization, proving the design is
> [compute/memory]-bound at an arithmetic intensity of [Y] ops/byte."

### "How would you improve it?"

> "I'd add double-buffering to the activation path and implement weight block
> prefetching to hide DMA latency. For a larger FPGA, I'd explore multi-bank
> BRAM with parallel DMAs to increase bandwidth."

---

## Part 5: Missing Test File (add to test_bsr_encoder.cpp)

```cpp
#include "compute/bsr_encoder.hpp"
#include <cassert>
#include <cstdio>

void test_dense_to_bsr() {
    // 28x28 dense matrix (4 blocks of 14x14)
    std::vector<int8_t> dense(28 * 28, 1);
    
    // Zero out one block
    for (int r = 0; r < 14; r++)
        for (int c = 0; c < 14; c++)
            dense[r * 28 + c] = 0;
    
    BSREncoder encoder;
    auto bsr = encoder.encode(dense.data(), 28, 28);
    
    assert(bsr.header.nnz_blocks == 3);  // 4 total - 1 zero = 3
    assert(bsr.row_ptr.size() == 3);     // 2 block rows + 1
    printf("BSR encoding test PASSED\n");
}

int main() {
    test_dense_to_bsr();
    return 0;
}
```

---

**Bottom Line:** Your RTL is solid. Focus C++ on:
1. Clean register access (`address_map.hpp`)
2. DMA management (`dma_controller.cpp`)  
3. **Performance analysis** (`performance.cpp`, `benchmark.cpp`)

The performance counters + roofline analysis are what will impress a memory architecture interviewer.
