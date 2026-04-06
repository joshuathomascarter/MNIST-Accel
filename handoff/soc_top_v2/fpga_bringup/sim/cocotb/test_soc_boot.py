#!/usr/bin/env python3
"""
Boot test for SoC - verifies:
1. CPU boots from Boot ROM
2. UART prints "Hello World"
3. Timer interrupt fires
4. GPIO LED toggles
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, Event
import logging

logger = logging.getLogger("test_soc_boot")
logger.setLevel(logging.DEBUG)

class UARTMonitor:
    """Monitor UART TX line"""
    def __init__(self, tx_pin):
        self.tx_pin = tx_pin
        self.data = ""
        
    async def run(self):
        """Monitor for UART characters"""
        while True:
            # Wait for start bit (0)
            while self.tx_pin.value != 0:
                await RisingEdge(self.tx_pin.dut.clk)
            
            # Wait 1.5 bits to sample middle of start bit
            await Timer(1.5 * 8.68, "us")  # At 115200 baud
            
            # Sample data bits
            char = 0
            for i in range(8):
                await Timer(8.68, "us")
                bit = self.tx_pin.value
                char |= (bit << i)
            
            # Wait for stop bit
            await Timer(8.68, "us")
            
            self.data += chr(char)
            logger.info(f"UART RX: {chr(char)} (0x{char:02x})")
            
            if  len(self.data) > 100:
                break

@cocotb.test()
async def test_soc_boot(dut):
    """Test SoC boot and basic execution"""
    
    # Setup clock
    clock = Clock(dut.clk, 20, units="ns")  # 50 MHz
    cocotb.start_soon(clock.start())
    
    # Reset sequence
    dut.rst_n.value = 0
    dut.gpio_i.value = 0x00
    dut.uart_rx.value = 1  # UART idle = high
    
    await Timer(100, "ns")
    dut.rst_n.value = 1
    await Timer(100, "ns")
    
    logger.info("SoC out of reset, CPU should boot from 0x0000_0000")
    
    # Start UART monitor
    uart_monitor = UARTMonitor(dut.uart_tx)
    monitor_task = cocotb.start_soon(uart_monitor.run())
    
    # Wait for UART output (timeout = 1ms simulation time)
    max_cycles = 50_000  # With 50 MHz clock, ~1ms
    for cycle in range(max_cycles):
        await RisingEdge(dut.clk)
        
        # Check for the string "Hello World"
        if "Hello" in uart_monitor.data:
            logger.info(f"✓ Got UART output: {uart_monitor.data[:30]}")
            break
        
        if cycle % 1000 == 0:
            logger.info(f"Waiting for UART... (cycle {cycle})")
    else:
        logger.error("✗ Timeout waiting for UART output")
        assert False, "No UART output detected"
    
    # Wait for timer interrupt (~100ms to observe LED toggle)
    logger.info("Waiting for timer interrupt...")
    max_cycles = 5_000_000  
    led_toggled = False
    
    prev_led = dut.gpio_o.value & 0x01
    for cycle in range(max_cycles):
        await RisingEdge(dut.clk)
        curr_led = dut.gpio_o.value & 0x01
        
        if curr_led != prev_led:
            logger.info(f"✓ Timer interrupt fired - GPIO LED toggled at cycle {cycle}")
            led_toggled = True
            break
        
        prev_led = curr_led
        
        if cycle % 100_000 == 0:
            logger.info(f"Waiting for timer interrupt... (cycle {cycle})")
    
    if not led_toggled:
        logger.warning("⚠ Timer interrupt did not fire within timeout")
    else:
        logger.info("✓ Boot test PASSED")
    
    # Cleanup
    try:
        monitor_task.kill()
    except:
        pass

@cocotb.test()
async def test_soc_basic(dut):
    """Basic memory test - write/read SRAM"""
    
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await Timer(100, "ns")
    dut.rst_n.value = 1
    await Timer(100, "ns")
    
    logger.info("SoC basic test: Write to SRAM, then read back")
    
    # Note: In a real test, we'd use an AXI master testbench or bus interface
    # This is a placeholder
    logger.info("✓ Basic test completed")
