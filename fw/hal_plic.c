/*
 * hal_plic.c — PLIC HAL driver
 * ==============================
 * RISC-V PLIC (Platform-Level Interrupt Controller).
 * Up to 32 external interrupt sources; 3-bit priority.
 */

#include "hal.h"

void hal_plic_init(void) {
    /* Disable all sources, threshold = 0 */
    REG32(PLIC_ENABLE) = 0;
    REG32(PLIC_THRESHOLD) = 0;
    /* Clear any pending claims */
    uint32_t id = REG32(PLIC_CLAIM);
    if (id)
        REG32(PLIC_CLAIM) = id;
}

void hal_plic_set_priority(uint32_t src, uint32_t pri) {
    if (src < 32)
        REG32(PLIC_PRIORITY(src)) = pri & 0x7;
}

void hal_plic_enable(uint32_t src) {
    if (src < 32)
        REG32(PLIC_ENABLE) |= (1u << src);
}

void hal_plic_disable(uint32_t src) {
    if (src < 32)
        REG32(PLIC_ENABLE) &= ~(1u << src);
}

void hal_plic_set_threshold(uint32_t thresh) {
    REG32(PLIC_THRESHOLD) = thresh & 0x7;
}

uint32_t hal_plic_claim(void) {
    return REG32(PLIC_CLAIM);
}

void hal_plic_complete(uint32_t src) {
    REG32(PLIC_CLAIM) = src;
}
