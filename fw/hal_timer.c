/*
 * hal_timer.c — Timer HAL driver
 * ================================
 * 64-bit mtime counter at TIMER_BASE.
 * Reads hi-lo-hi to avoid tearing on 32-bit bus.
 */

#include "hal.h"

uint64_t hal_timer_get(void) {
    uint32_t hi, lo, hi2;
    do {
        hi  = REG32(MTIME_HI);
        lo  = REG32(MTIME_LO);
        hi2 = REG32(MTIME_HI);
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

void hal_timer_set_compare(uint64_t cmp) {
    /* Write hi first with max value to prevent spurious interrupt */
    REG32(MTIMECMP_HI) = 0xFFFFFFFF;
    REG32(MTIMECMP_LO) = (uint32_t)(cmp & 0xFFFFFFFF);
    REG32(MTIMECMP_HI) = (uint32_t)(cmp >> 32);
}

void hal_timer_delay_us(uint32_t us) {
    /* Assumes 50 MHz system clock → 1 tick = 20 ns → 50 ticks/us */
    uint64_t target = hal_timer_get() + (uint64_t)us * 50;
    while (hal_timer_get() < target)
        ;
}
