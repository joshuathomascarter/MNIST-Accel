/*
 * hal_uart.c — UART HAL driver
 * =============================
 * TX: poll TX_FULL before writing.  RX: poll RX_EMPTY before reading.
 * Baud divisor = sys_clk / baud_rate  (written to BAUD_DIV register).
 */

#include "hal.h"

void hal_uart_init(uint32_t baud_div) {
    REG32(UART_BAUD_DIV) = baud_div;
    REG32(UART_CTRL) = UART_CTRL_TX_EN | UART_CTRL_RX_EN;
}

void hal_uart_putc(char c) {
    while (REG32(UART_STATUS) & UART_STATUS_TX_FULL)
        ;
    REG8(UART_TX_DATA) = (uint8_t)c;
}

char hal_uart_getc(void) {
    while (REG32(UART_STATUS) & UART_STATUS_RX_EMPTY)
        ;
    return (char)REG8(UART_RX_DATA);
}

bool hal_uart_rx_ready(void) {
    return !(REG32(UART_STATUS) & UART_STATUS_RX_EMPTY);
}

void hal_uart_puts(const char *s) {
    while (*s)
        hal_uart_putc(*s++);
}

void hal_uart_put_hex(uint32_t val) {
    static const char hex[] = "0123456789ABCDEF";
    hal_uart_puts("0x");
    for (int i = 28; i >= 0; i -= 4)
        hal_uart_putc(hex[(val >> i) & 0xF]);
}
