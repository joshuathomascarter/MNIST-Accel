/*
 * hal_eth.c — Ethernet HAL placeholder for soc_top_v2 bringup
 * ============================================================
 * The active soc_top_v2 top does not instantiate an Ethernet peripheral or
 * expose Ethernet IO pins, so these helpers intentionally report unsupported
 * rather than touching the reserved address window at ETH_BASE.
 */

#include "hal.h"

void hal_eth_init(void) {
}

int hal_eth_rx_poll(uint8_t *buf, uint32_t max_len) {
    (void)buf;
    (void)max_len;
    return -1;
}

void hal_eth_tx_send(const uint8_t *buf, uint32_t len) {
    (void)buf;
    (void)len;
}
