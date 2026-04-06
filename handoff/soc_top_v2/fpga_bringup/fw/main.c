// ============================================================================
// main.c — UART bringup loader for soc_top_v2 block-level dataflow
// ============================================================================

#include <stdint.h>
#include <stdbool.h>
#include "hal.h"
#include "hal_accel.h"

enum {
    CMD_PING = 0x00,
    CMD_MEM_WRITE_WORDS = 0x01,
    CMD_MEM_READ_WORDS = 0x02,
    CMD_TILE_LOAD = 0x10,
    CMD_TILE_COMPUTE = 0x11,
    CMD_TILE_STORE = 0x12,
    CMD_TILE_STATUS = 0x13,
    CMD_TILE_BARRIER_ALL = 0x14,
    CMD_ARGMAX = 0x20
};

enum {
    RESP_OK = 0x79,
    RESP_ERR = 0x1Fu
};

enum {
    ERR_BAD_CMD = 1u,
    ERR_BAD_TILE = 2u,
    ERR_BAD_COUNT = 3u
};

static void uart_putc(char c)
{
    while (REG32(UART_STATUS) & UART_STATUS_TX_FULL)
        ;
    REG8(UART_TX_DATA) = (uint8_t)c;
}

static char uart_getc(void)
{
    while (REG32(UART_STATUS) & UART_STATUS_RX_EMPTY)
        ;
    return (char)REG8(UART_RX_DATA);
}

static void uart_puts(const char *s)
{
    while (*s)
        uart_putc(*s++);
}

static void uart_put_u32(uint32_t value)
{
    uart_putc((char)(value & 0xFFu));
    uart_putc((char)((value >> 8) & 0xFFu));
    uart_putc((char)((value >> 16) & 0xFFu));
    uart_putc((char)((value >> 24) & 0xFFu));
}

static uint32_t uart_get_u32(void)
{
    uint32_t b0 = (uint8_t)uart_getc();
    uint32_t b1 = (uint8_t)uart_getc();
    uint32_t b2 = (uint8_t)uart_getc();
    uint32_t b3 = (uint8_t)uart_getc();
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static void send_ok(void)
{
    uart_putc((char)RESP_OK);
}

static void send_err(uint32_t code)
{
    uart_putc((char)RESP_ERR);
    uart_put_u32(code);
}

static bool tile_is_valid(uint32_t tile)
{
    return tile < ACCEL_NUM_TILES;
}

static void handle_mem_write_words(void)
{
    uint32_t addr = uart_get_u32();
    uint32_t words = uart_get_u32();

    if (words == 0u) {
        send_err(ERR_BAD_COUNT);
        return;
    }

    for (uint32_t index = 0; index < words; index++)
        REG32(addr + (index << 2)) = uart_get_u32();

    send_ok();
}

static void handle_mem_read_words(void)
{
    uint32_t addr = uart_get_u32();
    uint32_t words = uart_get_u32();

    if (words == 0u) {
        send_err(ERR_BAD_COUNT);
        return;
    }

    send_ok();
    for (uint32_t index = 0; index < words; index++)
        uart_put_u32(REG32(addr + (index << 2)));
}

static void handle_tile_load(void)
{
    uint32_t tile = uart_get_u32();
    uint32_t dram_addr = uart_get_u32();
    uint32_t word_count = uart_get_u32();
    uint32_t sp_off = uart_get_u32();

    if (!tile_is_valid(tile)) {
        send_err(ERR_BAD_TILE);
        return;
    }
    if (word_count == 0u || word_count > 0xFFFFu) {
        send_err(ERR_BAD_COUNT);
        return;
    }

    accel_load(tile, dram_addr, (uint16_t)word_count, sp_off);
    accel_wait_idle(tile);
    send_ok();
}

static void handle_tile_compute(void)
{
    uint32_t tile = uart_get_u32();
    uint32_t act_sp_off = uart_get_u32();
    uint32_t wgt_sp_off = uart_get_u32();
    uint32_t out_sp_off = uart_get_u32();

    if (!tile_is_valid(tile)) {
        send_err(ERR_BAD_TILE);
        return;
    }

    accel_compute_at(tile, act_sp_off, wgt_sp_off, out_sp_off);
    accel_wait_idle(tile);
    send_ok();
}

static void handle_tile_store(void)
{
    uint32_t tile = uart_get_u32();
    uint32_t dram_addr = uart_get_u32();
    uint32_t word_count = uart_get_u32();
    uint32_t sp_off = uart_get_u32();

    if (!tile_is_valid(tile)) {
        send_err(ERR_BAD_TILE);
        return;
    }
    if (word_count == 0u || word_count > 0xFFFFu) {
        send_err(ERR_BAD_COUNT);
        return;
    }

    accel_store(tile, dram_addr, (uint16_t)word_count, sp_off);
    accel_wait_idle(tile);
    send_ok();
}

static void handle_tile_status(void)
{
    uint32_t tile = uart_get_u32();

    if (!tile_is_valid(tile)) {
        send_err(ERR_BAD_TILE);
        return;
    }

    send_ok();
    uart_put_u32(accel_status(tile));
}

static void handle_tile_barrier_all(void)
{
    accel_barrier_all();
    send_ok();
}

static void handle_argmax(void)
{
    uint32_t addr = uart_get_u32();
    uint32_t count = uart_get_u32();
    int32_t best_value;
    uint32_t best_index = 0u;

    if (count == 0u) {
        send_err(ERR_BAD_COUNT);
        return;
    }

    best_value = (int32_t)REG32(addr);
    for (uint32_t index = 1; index < count; index++) {
        int32_t value = (int32_t)REG32(addr + (index << 2));
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }

    send_ok();
    uart_put_u32(best_index);
    uart_put_u32((uint32_t)best_value);
    REG32(GPIO_OUT) = (best_index & 0x0Fu) | 0xF0u;
}

int main(void)
{
    REG32(UART_CTRL) = UART_CTRL_TX_EN | UART_CTRL_RX_EN;
    REG32(GPIO_DIR) = 0xFFu;
    REG32(GPIO_OUT) = 0x00u;

    uart_puts("soc_top_v2 loader ready\r\n");
    uart_puts("use uncached aliases 0x60000000/0x70000000 for CPU preload\r\n");

    while (1) {
        uint8_t cmd = (uint8_t)uart_getc();

        switch (cmd) {
        case CMD_PING:
            send_ok();
            uart_put_u32(0x53545632u);
            break;
        case CMD_MEM_WRITE_WORDS:
            handle_mem_write_words();
            break;
        case CMD_MEM_READ_WORDS:
            handle_mem_read_words();
            break;
        case CMD_TILE_LOAD:
            handle_tile_load();
            break;
        case CMD_TILE_COMPUTE:
            handle_tile_compute();
            break;
        case CMD_TILE_STORE:
            handle_tile_store();
            break;
        case CMD_TILE_STATUS:
            handle_tile_status();
            break;
        case CMD_TILE_BARRIER_ALL:
            handle_tile_barrier_all();
            break;
        case CMD_ARGMAX:
            handle_argmax();
            break;
        default:
            send_err(ERR_BAD_CMD);
            break;
        }
    }

    return 0;
}

