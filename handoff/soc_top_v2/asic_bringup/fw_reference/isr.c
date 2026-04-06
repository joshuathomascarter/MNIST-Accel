/*
 * isr.c — Interrupt Service Routine framework
 * =============================================
 * Called from the trap vector in startup.S when an external interrupt fires.
 * Dispatches via PLIC claim/complete.
 */

#include "hal.h"

/* ---- IRQ source IDs (must match SoC PLIC wiring) --------------------- */
#define IRQ_UART_RX     1
#define IRQ_TIMER       2
#define IRQ_GPIO        3
#define IRQ_ETH_RX      4
#define IRQ_ACCEL_DONE  5

/* ---- Weak callbacks (override in application code) -------------------- */
__attribute__((weak)) void on_uart_rx(void)     { /* default: nop */ }
__attribute__((weak)) void on_timer(void)        { /* default: nop */ }
__attribute__((weak)) void on_gpio(void)         { /* default: nop */ }
__attribute__((weak)) void on_eth_rx(void)       { /* default: nop */ }
__attribute__((weak)) void on_accel_done(void)   { /* default: nop */ }

/* ---- External interrupt handler (called from trap vector) ------------- */
void isr_external(void) {
    uint32_t src = hal_plic_claim();
    if (src == 0)
        return;  /* spurious */

    switch (src) {
        case IRQ_UART_RX:     on_uart_rx();     break;
        case IRQ_TIMER:       on_timer();        break;
        case IRQ_GPIO:        on_gpio();         break;
        case IRQ_ETH_RX:      on_eth_rx();       break;
        case IRQ_ACCEL_DONE:  on_accel_done();   break;
        default:
            break;
    }

    hal_plic_complete(src);
}
