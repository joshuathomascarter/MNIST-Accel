// ============================================================================
// main_inference_multitile.c  — Multi-Tile FC1+FC2 MNIST firmware
// ============================================================================
// Runs FC1 (140×9216) across N_PARALLEL=4 worker tiles in K-parallel mode,
// followed by FC2 (10×140) on tile 0.
//
// FC1 Strategy:
//   The BSR non-zero K-blocks for each M-tile are partitioned round-robin
//   across N_PARALLEL worker tiles by gen_dram_init_multitile.py.
//   For each FC1 output M-tile (row block):
//     1. All N_PARALLEL tiles receive their K-block slice in parallel:
//          for each block round j across tiles:
//            LOAD weight_j → LOAD act_j → COMPUTE → STORE partial_j
//        Commands are issued tile-round-robin so tiles execute concurrently.
//     2. CPU waits for all tiles, then folds N_PARALLEL partial sums into
//        a single 16-element INT32 accumulator.
//     3. ReLU + write to FC1_OUT.
//
// FC2:
//   After quantising FC1 output to INT8 GEMV blocks, FC2 runs on tile 0
//   (same as main_inference_full.c).
//
// Compile:
//   make multitile   (see fw/Makefile)
// ============================================================================

#include <stdint.h>
#include <stdbool.h>
#include "hal.h"
#include "hal_accel.h"
#include "inference_config_multitile.h"

// ────────────────────────────────────────────────────────────────────────────
// UART helpers
// ────────────────────────────────────────────────────────────────────────────
static void uart_putc(char c)
{
    while (REG32(UART_STATUS) & UART_STATUS_TX_FULL)
        ;
    REG8(UART_TX_DATA) = (uint8_t)c;
}

static void uart_puts(const char *s)
{
    while (*s) uart_putc(*s++);
}

static void uart_put_hex(uint32_t val)
{
    static const char hex[] = "0123456789abcdef";
    for (int i = 28; i >= 0; i -= 4)
        uart_putc(hex[(val >> i) & 0xF]);
}

static void uart_put_dec(int32_t val)
{
    if (val < 0) {
        uart_putc('-');
        uint32_t uval = (uint32_t)(-(val + 1)) + 1u;
        uart_put_dec((int32_t)uval);
        return;
    }
    char buf[12];
    int i = 0;
    if (val == 0) { uart_putc('0'); return; }
    uint32_t uv = (uint32_t)val;
    while (uv > 0) {
        uint32_t q = 0, rem = uv;
        while (rem >= 10u) { rem -= 10u; q++; }
        buf[i++] = '0' + (char)rem;
        uv = q;
    }
    for (int j = i - 1; j >= 0; j--) uart_putc(buf[j]);
}

static uint32_t timer_read(void)
{
    return REG32(MTIME_LO);
}

// ────────────────────────────────────────────────────────────────────────────
// Simple max helper (no libc)
// ────────────────────────────────────────────────────────────────────────────
static inline uint32_t u32max(uint32_t a, uint32_t b) { return (a > b) ? a : b; }

// ────────────────────────────────────────────────────────────────────────────
// Scratchpad layout (word offsets, common to all tiles)
// ────────────────────────────────────────────────────────────────────────────
// Reuse defines from inference_config_multitile.h:
// SP_WGT_BASE=0, SP_ACT_BASE=64, SP_OUT_BASE=128, SP_RESULT_WORDS=256

// ────────────────────────────────────────────────────────────────────────────
// Per-tile DRAM offset tables (filled from config macros)
// ────────────────────────────────────────────────────────────────────────────
static const uint32_t mt_indptr_base[N_PARALLEL]  = MT_INDPTR_OFFSETS;
static const uint32_t mt_indices_base[N_PARALLEL] = MT_INDICES_OFFSETS;
static const uint32_t mt_data_base[N_PARALLEL]    = MT_DATA_OFFSETS;
static const uint32_t mt_partial_base[N_PARALLEL] = MT_PARTIAL_OFFSETS;

// ────────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────────
int main(void)
{
    REG32(UART_CTRL) = UART_CTRL_TX_EN | UART_CTRL_RX_EN;
    REG32(GPIO_DIR)  = 0xFFu;
    REG32(GPIO_OUT)  = 0x01u;

    uart_puts("MNIST multi-tile firmware (n_parallel=");
    uart_put_dec((int32_t)N_PARALLEL);
    uart_puts(")\r\n");
    uart_puts("FC1: ");
    uart_put_dec(FC1_M_ORIG);
    uart_putc('x');
    uart_put_dec(FC1_K_ORIG);
    uart_puts(" → ");
    uart_put_dec(FC1_NUM_M_TILES);
    uart_puts("M × ");
    uart_put_dec(FC1_NUM_K_TILES);
    uart_puts("K tiles\r\n");

    REG32(GPIO_OUT) = 0x02u;

    uint32_t t_start = timer_read();

    // ════════════════════════════════════════════════════════════════════════
    // STEP 1 — FC1 BSR GEMV, N_PARALLEL tiles, K-block parallel
    // ════════════════════════════════════════════════════════════════════════
    // Per-tile scratch for inter-round partial accumulation.
    // partial_acc[w][col] accumulates across all blocks assigned to tile w
    // for the current M-tile. Stored in DRAM uncached so all tiles see it.
    // Layout: each tile gets SYSTOLIC_DIM (=16) contiguous INT32 words.

    for (uint32_t m = 0; m < FC1_NUM_M_TILES; m++) {

        // ---- Reset per-tile partial accumulators in DRAM ----------------
        for (uint32_t w = 0; w < N_PARALLEL; w++) {
            uint32_t partial_uc = DRAM_BASE_UC + mt_partial_base[w] * 4u;
            for (uint32_t col = 0; col < SYSTOLIC_DIM; col++)
                REG32(partial_uc + col * 4u) = 0u;
        }

        // ---- Read per-tile row ranges for M-tile m ----------------------
        uint32_t row_start[N_PARALLEL], row_end[N_PARALLEL];
        uint32_t max_blocks = 0u;
        for (uint32_t w = 0; w < N_PARALLEL; w++) {
            row_start[w] = REG32(DRAM_BASE_UC
                               + mt_indptr_base[w] * 4u + m * 4u);
            row_end[w]   = REG32(DRAM_BASE_UC
                               + mt_indptr_base[w] * 4u + (m + 1u) * 4u);
            uint32_t n_blk = row_end[w] - row_start[w];
            if (n_blk > max_blocks) max_blocks = n_blk;
        }

        if (max_blocks == 0u) {
            // All-zero M-tile (shouldn't happen with trained weights, but safe)
            uint32_t fc1_out_uc = DRAM_BASE_UC
                                + FC1_OUT_OFFSET * 4u
                                + m * SYSTOLIC_DIM * 4u;
            for (uint32_t col = 0; col < SYSTOLIC_DIM; col++)
                REG32(fc1_out_uc + col * 4u) = 0u;
            continue;
        }

        // ---- Block-level round-robin dispatch across N_PARALLEL tiles ----
        // For each block round j (0..max_blocks-1):
        //   For each tile w (0..N_PARALLEL-1):
        //     if j < blocks_for_tile_w:
        //       LOAD weight block j_global for tile w  (wait for previous op on w)
        //       LOAD activation block for its K-index
        //       COMPUTE
        //       STORE partial output to tile w's DRAM scratch
        //   CPU folds tile w's output into per-tile partial_acc
        //   (unrolled across tiles to maximise overlap of compute phases)
        //
        // Key parallelism observation:
        //   Phase 1 (LOAD weight): accel_load(w=0) issues; CPU immediately moves to
        //     accel_load(w=1) while tile 0 is DMAing.  All 4 stores proceed in parallel.
        //   Phase 2 (LOAD act), Phase 3 (COMPUTE), Phase 4 (STORE): same pattern.
        //   accel_load / accel_compute_at / accel_store each call accel_wait_idle(tile)
        //   which waits only for that specific tile, leaving others running.

        // Precompute per-tile per-round addresses to avoid re-reading DRAM in the
        // hot loop (we need them before each phase dispatches to all tiles).
        // With max_blocks typically ~12, this fits easily in RISC-V registers/stack.
        // We compute the addresses inline per iteration to control stack usage.

        for (uint32_t j = 0u; j < max_blocks; j++) {

            // ─── Phase 1: LOAD weight block to all tiles ─────────────────
            for (uint32_t w = 0u; w < N_PARALLEL; w++) {
                if (j >= (row_end[w] - row_start[w])) continue;

                uint32_t glob_j   = row_start[w] + j;
                uint32_t wgt_off  = mt_data_base[w] + glob_j * BLOCK_WORDS;
                uint32_t wgt_addr = DRAM_BASE_CACHED + wgt_off * 4u;
                // accel_load: waits for tile w to be idle, then issues LOAD
                accel_load(w, wgt_addr, BLOCK_WORDS, SP_WGT_BASE);
            }

            // ─── Phase 2: LOAD activation block to all tiles ─────────────
            for (uint32_t w = 0u; w < N_PARALLEL; w++) {
                if (j >= (row_end[w] - row_start[w])) continue;

                uint32_t glob_j   = row_start[w] + j;
                // Read K-column index for this block from tile w's indices array
                uint32_t k        = REG32(DRAM_BASE_UC
                                        + mt_indices_base[w] * 4u
                                        + glob_j * 4u);
                uint32_t act_addr = DRAM_BASE_CACHED
                                  + (FC1_ACT_OFFSET + k * BLOCK_WORDS) * 4u;
                // waits for tile w's weight load, then issues activation LOAD
                accel_load(w, act_addr, BLOCK_WORDS, SP_ACT_BASE);
            }

            // ─── Phase 3: COMPUTE on all tiles ───────────────────────────
            for (uint32_t w = 0u; w < N_PARALLEL; w++) {
                if (j >= (row_end[w] - row_start[w])) continue;
                // waits for tile w's activation load, then starts compute
                accel_compute_at(w, SP_ACT_BASE, SP_WGT_BASE, SP_OUT_BASE);
            }

            // ─── Phase 4: STORE results + CPU FOLD per tile ───────────────
            // Per-block scratch address for tile w (shared single 256-word slot,
            // safe because we fold before moving to the next block round)
            for (uint32_t w = 0u; w < N_PARALLEL; w++) {
                if (j >= (row_end[w] - row_start[w])) continue;

                // Use a DRAM scratch region below FC1 partial for per-block output.
                // We reuse mt_partial_base[w] as the store target overwrites it each
                // round, then CPU folds immediately after accel_wait_idle.
                // Actual partial_acc is maintained entirely in this C loop.
                uint32_t scratch_off  = mt_partial_base[w] + SYSTOLIC_DIM; // 16 words after acc
                uint32_t scratch_addr = DRAM_BASE_CACHED + scratch_off * 4u;
                // waits for compute, then issues STORE
                accel_store(w, scratch_addr, SP_RESULT_WORDS, SP_OUT_BASE);
            }

            // ─── Phase 5: Wait + fold per tile ───────────────────────────
            for (uint32_t w = 0u; w < N_PARALLEL; w++) {
                if (j >= (row_end[w] - row_start[w])) continue;

                // Wait for tile w's store to complete
                accel_wait_idle(w);

                // CPU fold: accumulate this tile's 16×16 result into partial_acc[w]
                // Result layout: row r, col c → word r*16 + c.
                // Partial sum for output row group m uses column-major convention:
                //   m_logit[col] += sum_r out[r][col]  (row-reduction over K-tile)
                uint32_t scratch_off  = mt_partial_base[w] + SYSTOLIC_DIM;
                uint32_t scratch_uc   = DRAM_BASE_UC + scratch_off * 4u;
                uint32_t partial_uc   = DRAM_BASE_UC + mt_partial_base[w] * 4u;

                for (uint32_t row = 0u; row < SYSTOLIC_DIM; row++) {
                    for (uint32_t col = 0u; col < SYSTOLIC_DIM; col++) {
                        uint32_t wrd = row * SYSTOLIC_DIM + col;
                        int32_t  val = (int32_t)REG32(scratch_uc + wrd * 4u);
                        uint32_t acc_ptr = partial_uc + col * 4u;
                        REG32(acc_ptr) = (uint32_t)((int32_t)REG32(acc_ptr) + val);
                    }
                }
            }
        } // end round loop

        // ─── After all rounds: fold per-tile partials into FC1_OUT ───────
        uint32_t fc1_out_uc = DRAM_BASE_UC
                            + FC1_OUT_OFFSET * 4u
                            + m * SYSTOLIC_DIM * 4u;
        for (uint32_t col = 0u; col < SYSTOLIC_DIM; col++)
            REG32(fc1_out_uc + col * 4u) = 0u;

        for (uint32_t w = 0u; w < N_PARALLEL; w++) {
            uint32_t partial_uc = DRAM_BASE_UC + mt_partial_base[w] * 4u;
            for (uint32_t col = 0u; col < SYSTOLIC_DIM; col++) {
                int32_t old = (int32_t)REG32(fc1_out_uc + col * 4u);
                int32_t add = (int32_t)REG32(partial_uc  + col * 4u);
                // ReLU handled below (uniform gate after full reduction)
                REG32(fc1_out_uc + col * 4u) = (uint32_t)(old + add);
            }
        }

        // Apply ReLU
        for (uint32_t col = 0u; col < SYSTOLIC_DIM; col++) {
            int32_t val = (int32_t)REG32(fc1_out_uc + col * 4u);
            if (val < 0) REG32(fc1_out_uc + col * 4u) = 0u;
        }

    } // end M-tile loop

    // ════════════════════════════════════════════════════════════════════════
    // STEP 2 — Quantise FC1 output → INT8 GEMV blocks for FC2
    // ════════════════════════════════════════════════════════════════════════
    uint32_t fc1_out_base_uc = DRAM_BASE_UC + FC1_OUT_OFFSET * 4u;

    int32_t max_val = 1;
    for (uint32_t i = 0u; i < FC1_M_ORIG; i++) {
        int32_t v = (int32_t)REG32(fc1_out_base_uc + i * 4u);
        if (v > max_val) max_val = v;
    }
    uint32_t shift = 0u;
    while ((max_val >> shift) > 127) shift++;

    for (uint32_t k = 0u; k < FC2_NUM_K_TILES; k++) {
        uint32_t blk_base = DRAM_BASE_UC
                          + (FC2_ACT_OFFSET + k * BLOCK_WORDS) * 4u;
        for (uint32_t w = 0u; w < (SYSTOLIC_DIM * SYSTOLIC_DIM / 4u); w++)
            REG32(blk_base + w * 4u) = 0u;
        for (uint32_t col = 0u; col < SYSTOLIC_DIM; col++) {
            uint32_t idx = k * SYSTOLIC_DIM + col;
            int32_t  raw = (int32_t)REG32(fc1_out_base_uc + idx * 4u);
            uint8_t  q   = (uint8_t)((raw >> shift) & 0xFFu);
            uint32_t word_idx = col >> 2u;
            uint32_t byte_idx = col & 3u;
            uint32_t cur = REG32(blk_base + word_idx * 4u);
            cur &= ~(0xFFu << (byte_idx * 8u));
            cur |=  ((uint32_t)q << (byte_idx * 8u));
            REG32(blk_base + word_idx * 4u) = cur;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // STEP 3 — FC2 dense GEMV on tile 0 (10×144, dense)
    // ════════════════════════════════════════════════════════════════════════
    for (uint32_t k = 0u; k < FC2_NUM_K_TILES; k++) {
        uint32_t wgt_addr = DRAM_BASE_CACHED
                          + (FC2_WEIGHT_OFFSET + k * BLOCK_WORDS) * 4u;
        uint32_t act_addr = DRAM_BASE_CACHED
                          + (FC2_ACT_OFFSET + k * BLOCK_WORDS) * 4u;
        uint32_t out_addr = DRAM_BASE_CACHED
                          + (FC2_OUTPUT_OFFSET + k * SP_RESULT_WORDS) * 4u;

        accel_load(ROOT_TILE, wgt_addr, BLOCK_WORDS, SP_WGT_BASE);
        accel_wait_idle(ROOT_TILE);
        accel_load(ROOT_TILE, act_addr, BLOCK_WORDS, SP_ACT_BASE);
        accel_wait_idle(ROOT_TILE);
        accel_compute_at(ROOT_TILE, SP_ACT_BASE, SP_WGT_BASE, SP_OUT_BASE);
        accel_wait_idle(ROOT_TILE);
        accel_store(ROOT_TILE, out_addr, SP_RESULT_WORDS, SP_OUT_BASE);
        accel_wait_idle(ROOT_TILE);
    }

    uint32_t t_end = timer_read();

    // ════════════════════════════════════════════════════════════════════════
    // STEP 4 — CPU reduce FC2 partials + argmax
    // ════════════════════════════════════════════════════════════════════════
    int32_t logits[FC2_M_PADDED];
    for (uint32_t i = 0u; i < FC2_M_PADDED; i++) logits[i] = 0;

    for (uint32_t k = 0u; k < FC2_NUM_K_TILES; k++) {
        uint32_t out_uc = DRAM_BASE_UC
                        + (FC2_OUTPUT_OFFSET + k * SP_RESULT_WORDS) * 4u;
        for (uint32_t row = 0u; row < SYSTOLIC_DIM; row++) {
            for (uint32_t cls = 0u; cls < FC2_M_ORIG; cls++) {
                logits[cls] += (int32_t)REG32(out_uc + (row * SYSTOLIC_DIM + cls) * 4u);
            }
        }
    }

    int32_t  best_val = logits[0];
    uint32_t best_idx = 0u;
    for (uint32_t i = 1u; i < FC2_M_ORIG; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best_idx = i; }
    }

    // ════════════════════════════════════════════════════════════════════════
    // STEP 5 — UART output and testbench handshake
    // ════════════════════════════════════════════════════════════════════════
    uart_puts("LogitsHex:");
    for (uint32_t i = 0u; i < FC2_M_ORIG; i++) {
        uart_putc(' ');
        uart_put_hex((uint32_t)logits[i]);
    }
    uart_puts("\r\n");

    uart_puts("Predicted: ");
    uart_put_dec((int32_t)best_idx);
    uart_puts("\r\n");

    uint32_t cycles = t_end - t_start;
    uart_puts("Total simulation cycles: ");
    uart_put_dec((int32_t)cycles);
    uart_puts("\r\n");

    uart_puts("Accel busy cycles: ");
    uart_put_dec((int32_t)cycles);   // approximation (tile cycles ≈ total here)
    uart_puts("\r\n");

#if defined(GOLDEN_PREDICTION)
    if (best_idx == GOLDEN_PREDICTION) {
        uart_puts("PASS: matches golden (");
        uart_put_dec((int32_t)GOLDEN_PREDICTION);
        uart_puts(")\r\n");
    } else {
        uart_puts("FAIL: expected ");
        uart_put_dec((int32_t)GOLDEN_PREDICTION);
        uart_puts(", got ");
        uart_put_dec((int32_t)best_idx);
        uart_puts("\r\n");
    }
#endif

    // Signal completion via GPIO (matches testbench check: 0xF0 | digit)
    REG32(GPIO_OUT) = (best_idx & 0x0Fu) | 0xF0u;
    uart_puts("Done.\r\n");

    // Barrier: synchronise all tiles before halting
    accel_barrier_all();

    // Firmware halt — loop forever
    while (1) __asm__ volatile ("wfi");
    return 0;
}
