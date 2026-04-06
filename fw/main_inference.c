// ============================================================================
// main_inference.c — Self-contained MNIST inference firmware
// ============================================================================
// Runs a single-layer FC2 (10×140) classification on tile 0:
//   1. For each K-tile (9 tiles): DMA weight block + activation vector
//      into tile scratchpad, run OP_COMPUTE, accumulate
//   2. Read back 16 INT32 accumulators from scratchpad via OP_STORE
//   3. Argmax over the first 10 values (= 10 output classes)
//   4. Print result over UART
//
// The conv layers + fc1 are pre-computed in Python; only fc2 runs on HW.
// This proves the complete data path:
//   DRAM → DRAM controller → NoC → tile DMA → scratchpad → systolic 16×16
//   → scratchpad → tile DMA → NoC → DRAM → CPU readback → UART
// ============================================================================

#include <stdint.h>
#include <stdbool.h>
#include "hal.h"
#include "hal_accel.h"
#include "inference_config.h"

// ---- UART helpers (same as main.c) ----------------------------------------
static void uart_putc(char c)
{
    while (REG32(UART_STATUS) & UART_STATUS_TX_FULL)
        ;
    REG8(UART_TX_DATA) = (uint8_t)c;
}

static void uart_puts(const char *s)
{
    while (*s)
        uart_putc(*s++);
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
        // Manual negate to avoid overflow for INT32_MIN
        uint32_t uval = (uint32_t)(-(val + 1)) + 1u;
        uart_put_dec((int32_t)uval);
        return;
    }
    char buf[12];
    int i = 0;
    if (val == 0) {
        uart_putc('0');
        return;
    }
    uint32_t uv = (uint32_t)val;
    while (uv > 0) {
        // Manual mod/div by 10 using shifts (avoid __divsi3/__modsi3)
        // q = uv / 10 ≈ (uv * 0xCCCCCCCD) >> 35 ... too complex
        // Just use subtract loop instead
        uint32_t q = 0;
        uint32_t rem = uv;
        // Simple repeated subtraction for small values
        while (rem >= 10u) { rem -= 10u; q++; }
        buf[i++] = '0' + (char)rem;
        uv = q;
    }
    for (int j = i - 1; j >= 0; j--)
        uart_putc(buf[j]);
}

// ---- Timer helpers --------------------------------------------------------
static uint32_t timer_read(void)
{
    return REG32(MTIME_LO);
}

// ---- Inference entry point ------------------------------------------------
//
// FC2 layer:  W[16×144] × x[144×1] = y[16×1]
// Tiled as:   1 M-tile × 9 K-tiles
//
// For each k_tile (0..8):
//   - Weight block: 16×16 INT8 at DRAM word offset
//       FC2_WEIGHT_OFFSET + k_tile * BLOCK_WORDS
//   - Activation block: 16×16 INT8 at DRAM word offset
//       ACT_OFFSET + k_tile * ACT_BLOCK_WORDS
//     Row 0 carries the 16 FC2 input values; rows 1..15 are zero.
//   - Scratchpad layout per tile-op:
//       SP_WGT_BASE  = 0    (16×16 INT8 = 64 words)
//       SP_ACT_BASE  = 64   (16×16 INT8 = 64 words)
//       SP_OUT_BASE  = 128  (16×16 INT32 = 256 words — only first 16 used)
//
// After each K-tile, the tile emits a 16×16 partial-product matrix. The CPU
// reduces those partials across rows and across all 9 K-tiles to produce the
// 10 FC2 logits.

#define TILE_ID         0u

// Scratchpad word offsets
#define SP_WGT_BASE     0u
#define SP_ACT_BASE     64u
#define SP_OUT_BASE     128u

// Number of INT32 accumulator words to store (16 rows × 16 cols = 256)
// But we only need the first column (16 values) for a vector GEMV.
// The tile stores all 256 accumulators regardless.
#define SP_RESULT_WORDS 256u

int main(void)
{
    // ---- Initialise peripherals ----
    REG32(UART_CTRL) = UART_CTRL_TX_EN | UART_CTRL_RX_EN;
    REG32(GPIO_DIR) = 0xFFu;
    REG32(GPIO_OUT) = 0x01u;  // LED0 = booted

    uart_puts("MNIST inference firmware v1\r\n");
    uart_puts("FC2: ");
    uart_put_dec(FC2_M_PADDED);
    uart_putc('x');
    uart_put_dec(FC2_K_PADDED);
    uart_puts(" (");
    uart_put_dec(FC2_NUM_M_TILES);
    uart_puts("m x ");
    uart_put_dec(FC2_NUM_K_TILES);
    uart_puts("k tiles)\r\n");

    uint32_t t_start = timer_read();
    int32_t logits[FC2_M_PADDED];

    for (uint32_t i = 0; i < FC2_M_PADDED; i++)
        logits[i] = 0;

    // ---- Signal busy ----
    REG32(GPIO_OUT) = 0x02u;  // LED1 = inference running

    // ---- Run tiled FC2: for each K-tile, load weight + act, compute ----
    for (uint32_t k = 0; k < FC2_NUM_K_TILES; k++) {
        // Weight block DRAM address (word offset → byte address in cached DRAM)
        // DMA goes through crossbar which only routes 0x4xxx to DRAM slave.
        uint32_t wgt_dram_addr = DRAM_BASE_CACHED +
                                 (FC2_WEIGHT_OFFSET + k * BLOCK_WORDS) * 4u;
        // Activation block DRAM address
        uint32_t act_dram_addr = DRAM_BASE_CACHED +
                     (ACT_OFFSET + k * ACT_BLOCK_WORDS) * 4u;

        // 1. DMA weight block (64 words) into scratchpad at SP_WGT_BASE
        accel_load(TILE_ID, wgt_dram_addr, BLOCK_WORDS, SP_WGT_BASE);
        accel_wait_idle(TILE_ID);

        // 2. DMA activation block (64 words) into scratchpad at SP_ACT_BASE
        accel_load(TILE_ID, act_dram_addr, ACT_BLOCK_WORDS, SP_ACT_BASE);
        accel_wait_idle(TILE_ID);

        // 3. Compute: systolic 16×16 multiply-accumulate
        accel_compute_at(TILE_ID, SP_ACT_BASE, SP_WGT_BASE, SP_OUT_BASE);
        accel_wait_idle(TILE_ID);

        // 4. Store the 16x16 partial-product tile for this K tile
        uint32_t out_tile_offset = OUTPUT_OFFSET + k * SP_RESULT_WORDS;
        uint32_t out_tile_addr_dma = DRAM_BASE_CACHED + out_tile_offset * 4u;
        accel_store(TILE_ID, out_tile_addr_dma, SP_RESULT_WORDS, SP_OUT_BASE);
        accel_wait_idle(TILE_ID);
    }

    uint32_t t_end = timer_read();

    // ---- CPU readback + reduction + argmax ----
    // Each stored tile is a 16x16 matrix of partial products. Summing over the
    // 16 rows yields the partial logits for this K tile; summing all K tiles
    // yields the final 10 logits.
    for (uint32_t k = 0; k < FC2_NUM_K_TILES; k++) {
        uint32_t out_tile_addr = DRAM_BASE_UC + (OUTPUT_OFFSET + k * SP_RESULT_WORDS) * 4u;
        for (uint32_t row = 0; row < SYSTOLIC_DIM; row++) {
            for (uint32_t cls = 0; cls < FC2_M_ORIG; cls++) {
                uint32_t word_off = row * SYSTOLIC_DIM + cls;
                logits[cls] += (int32_t)REG32(out_tile_addr + word_off * 4u);
            }
        }
    }

    int32_t best_val = logits[0];
    uint32_t best_idx = 0;

    for (uint32_t i = 0; i < FC2_M_ORIG; i++) {
        int32_t val = logits[i];
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    uart_puts("LogitsHex:");
    for (uint32_t i = 0; i < FC2_M_ORIG; i++) {
        uart_putc(' ');
        uart_put_hex((uint32_t)logits[i]);
    }
    uart_puts("\r\n");

    // ---- Report ----
    uart_puts("Predicted: ");
    uart_put_dec((int32_t)best_idx);
    uart_puts("\r\n");

    if (TRUE_LABEL == 0xFFFFFFFFu) {
        uart_puts("True label: n/a\r\n");
    } else {
        uart_puts("True label: ");
        uart_put_dec((int32_t)TRUE_LABEL);
        uart_puts("\r\n");
    }

    if (best_idx == GOLDEN_PREDICTION) {
        uart_puts("PASS: matches golden\r\n");
    } else {
        uart_puts("FAIL: expected ");
        uart_put_dec((int32_t)GOLDEN_PREDICTION);
        uart_puts(" got ");
        uart_put_dec((int32_t)best_idx);
        uart_puts("\r\n");
    }

    uint32_t cycles = t_end - t_start;
    uart_puts("Cycles: ");
    uart_put_hex(cycles);
    uart_puts("\r\n");

    // ---- Signal done ----
    REG32(GPIO_OUT) = (best_idx & 0x0Fu) | 0xF0u;

    uart_puts("Done.\r\n");

    // Halt
    while (1)
        ;

    return 0;
}
