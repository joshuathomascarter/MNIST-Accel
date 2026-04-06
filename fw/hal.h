/*
 * hal.h — Hardware Abstraction Layer header for RISC-V SoC
 * =========================================================
 * Memory map:
 *   ROM   0x0000_0000   SRAM  0x1000_0000
 *   UART  0x2000_0000   GPIO  0x2002_0000
 *   TIMER 0x2001_0000   PLIC  0x2003_0000
 *   ETH   0x2004_0000   ACCEL 0x3000_0000  (ETH reserved / unimplemented)
 *   DRAM  0x4000_0000   DRAM_UC 0x6000_0000
 *                        SRAM_UC 0x7000_0000
 */

#ifndef HAL_H
#define HAL_H

#include <stdint.h>
#include <stdbool.h>

/* ---- Register access helpers ------------------------------------------ */
#define REG32(addr)  (*(volatile uint32_t *)(uintptr_t)(addr))
#define REG8(addr)   (*(volatile uint8_t  *)(uintptr_t)(addr))

/* ---- Base addresses --------------------------------------------------- */
#define ROM_BASE     0x00000000u
#define SRAM_BASE    0x10000000u
#define UART_BASE    0x20000000u
#define TIMER_BASE   0x20010000u
#define GPIO_BASE    0x20020000u
#define PLIC_BASE    0x20030000u
#define ETH_BASE     0x20040000u
#define ACCEL_BASE   0x30000000u
#define DRAM_BASE    0x40000000u

/* ---- Uncached alias windows for CPU preload/readback ------------------ */
#define DRAM_BASE_UC 0x60000000u
#define SRAM_BASE_UC 0x70000000u

/* ===== UART ============================================================ */
#define UART_TX_DATA    (UART_BASE + 0x00)
#define UART_RX_DATA    (UART_BASE + 0x04)
#define UART_STATUS     (UART_BASE + 0x08)
#define UART_CTRL       (UART_BASE + 0x0C)
#define UART_BAUD_DIV   (UART_BASE + 0x10)

/* Status bits */
#define UART_STATUS_TX_FULL   (1u << 0)
#define UART_STATUS_RX_EMPTY  (1u << 1)
#define UART_STATUS_TX_EMPTY  (1u << 2)

/* Control bits */
#define UART_CTRL_TX_EN   (1u << 0)
#define UART_CTRL_RX_EN   (1u << 1)
#define UART_CTRL_IRQ_EN  (1u << 2)

void     hal_uart_init(uint32_t baud_div);
void     hal_uart_putc(char c);
char     hal_uart_getc(void);
bool     hal_uart_rx_ready(void);
void     hal_uart_puts(const char *s);
void     hal_uart_put_hex(uint32_t val);

/* ===== TIMER =========================================================== */
#define MTIME_LO     (TIMER_BASE + 0x00)
#define MTIME_HI     (TIMER_BASE + 0x04)
#define MTIMECMP_LO  (TIMER_BASE + 0x08)
#define MTIMECMP_HI  (TIMER_BASE + 0x0C)
#define TIMER_CTRL   (TIMER_BASE + 0x10)

uint64_t hal_timer_get(void);
void     hal_timer_set_compare(uint64_t cmp);
void     hal_timer_delay_us(uint32_t us);

/* ===== GPIO ============================================================ */
#define GPIO_DIR   (GPIO_BASE + 0x00)
#define GPIO_OUT   (GPIO_BASE + 0x04)
#define GPIO_IN    (GPIO_BASE + 0x08)

void     hal_gpio_set_dir(uint32_t mask);
void     hal_gpio_write(uint32_t val);
uint32_t hal_gpio_read(void);
void     hal_gpio_set_pin(uint8_t pin);
void     hal_gpio_clear_pin(uint8_t pin);
bool     hal_gpio_read_pin(uint8_t pin);

/* ===== PLIC ============================================================ */
#define PLIC_PRIORITY(src)   (PLIC_BASE + 0x000000u + (src)*4)
#define PLIC_PENDING         (PLIC_BASE + 0x001000u)
#define PLIC_ENABLE          (PLIC_BASE + 0x002000u)
#define PLIC_THRESHOLD       (PLIC_BASE + 0x200000u)
#define PLIC_CLAIM           (PLIC_BASE + 0x200004u)

void     hal_plic_init(void);
void     hal_plic_set_priority(uint32_t src, uint32_t pri);
void     hal_plic_enable(uint32_t src);
void     hal_plic_disable(uint32_t src);
void     hal_plic_set_threshold(uint32_t thresh);
uint32_t hal_plic_claim(void);
void     hal_plic_complete(uint32_t src);

/* ===== ETH ============================================================= */
#define ETH_STATUS    (ETH_BASE + 0x00)
#define ETH_CTRL      (ETH_BASE + 0x04)
#define ETH_RX_ADDR   (ETH_BASE + 0x08)
#define ETH_RX_LEN    (ETH_BASE + 0x0C)
#define ETH_TX_ADDR   (ETH_BASE + 0x10)
#define ETH_TX_LEN    (ETH_BASE + 0x14)

void     hal_eth_init(void);
int      hal_eth_rx_poll(uint8_t *buf, uint32_t max_len);
void     hal_eth_tx_send(const uint8_t *buf, uint32_t len);

/* ===== ISR ============================================================= */
void     isr_external(void);

#endif /* HAL_H */
