/*
 * hal_gpio.c — GPIO HAL driver
 * ==============================
 * 32-bit GPIO with direction register.
 *   DIR:  1 = output, 0 = input
 *   OUT:  drive value (only effective on output pins)
 *   IN:   read current pin state
 */

#include "hal.h"

void hal_gpio_set_dir(uint32_t mask) {
    REG32(GPIO_DIR) = mask;
}

void hal_gpio_write(uint32_t val) {
    REG32(GPIO_OUT) = val;
}

uint32_t hal_gpio_read(void) {
    return REG32(GPIO_IN);
}

void hal_gpio_set_pin(uint8_t pin) {
    REG32(GPIO_OUT) |= (1u << pin);
}

void hal_gpio_clear_pin(uint8_t pin) {
    REG32(GPIO_OUT) &= ~(1u << pin);
}

bool hal_gpio_read_pin(uint8_t pin) {
    return (REG32(GPIO_IN) >> pin) & 1u;
}
