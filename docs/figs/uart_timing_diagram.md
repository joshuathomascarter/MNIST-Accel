# UART Protocol Timing Diagram

```
ACCEL-v1 UART Communication Protocol
===================================

Packet Structure:
┌──────┬──────┬────────┬─────────┬──────────────────┬─────┐
│ SYNC0│ SYNC1│ LENGTH │ COMMAND │     PAYLOAD      │ CRC │
│ 0xA5 │ 0x5A │   N    │  CMD    │     N bytes      │ CRC8│
└──────┴──────┴────────┴─────────┴──────────────────┴─────┘
  1B     1B      1B       1B          0-255B         1B

═══════════════════════════════════════════════════════════════════

Timing Diagram: CSR Write Operation
═══════════════════════════════════════════════════════════════════

Host Side (Python)                Hardware Side (Verilog)
━━━━━━━━━━━━━━━                     ━━━━━━━━━━━━━━━━━━━━━━━━

write_csr(0x04, 0x00000008)

1. Packet Generation:
   ┌─────────────────────┐
   │ make_packet()       │
   │ cmd=0x01 (WRITE)    │         
   │ payload=addr+data   │
   └─────────────────────┘

2. UART Transmission:                  UART Reception:
   Time →                              Time →
   
   0xA5 ┌─┐     ┌─┐     ┌─┐            0xA5 ┌─┐     ┌─┐     ┌─┐
        └─┘─────┘─┘─────┘─┘───────────▶     └─┘─────┘─┘─────┘─┘
   
   0x5A ┌─┐   ┌─┐ ┌─┐ ┌─┐              0x5A ┌─┐   ┌─┐ ┌─┐ ┌─┐
        └─┘───┘─┘─┘─┘─┘─┘───────────────▶   └─┘───┘─┘─┘─┘─┘─┘
   
   0x08 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x08 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   
   0x01 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x01 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   
   Payload (8 bytes):                       Payload (8 bytes):
   0x04 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x04 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   0x08 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x08 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐                0x00 ┌─┐ ┌─┐ ┌─┐ ┌─┐
        └─┘─┘─┘─┘─┘─┘─┘─────────────────▶   └─┘─┘─┘─┘─┘─┘─┘
   
   CRC ┌─┐ ┌─┐ ┌─┐ ┌─┐                 CRC ┌─┐ ┌─┐ ┌─┐ ┌─┐
       └─┘─┘─┘─┘─┘─┘─┘──────────────────▶  └─┘─┘─┘─┘─┘─┘─┘

3. Packet Processing:                       3. Hardware Processing:
   ┌─────────────────────┐                     ┌─────────────────────┐
   │ send_packet()       │                     │ UART RX ISR         │
   │ • Write to serial   │                     │ • Frame detection   │
   │ • Flush buffer      │                     │ • CRC validation    │
   └─────────────────────┘                     │ • Command decode    │
                                               └─────────────────────┘
                                                         │
4. Wait for Response:                                    ▼
   ┌─────────────────────┐                     ┌─────────────────────┐
   │ recv_packet()       │                     │ CSR Write Handler   │
   │ • Poll serial port  │                     │ addr = 0x00000004   │
   │ • Timeout handling  │◀ ─ ─ ─ ─ ─ ─ ─ ─ ─ │ data = 0x00000008   │
   └─────────────────────┘                     │ csr_regs[4] = data  │
                                               └─────────────────────┘

═══════════════════════════════════════════════════════════════════

Complete GEMM Operation Timeline
═══════════════════════════════════════════════════════════════════

Time →   0      1      2      3      4      5      6      7      8
Host:    │ CFG  │ WGT  │ ACT  │START │ POLL │ POLL │ POLL │ READ │ ACK │
         │ CSRs │ Data │ Data │ OP   │STATUS│STATUS│STATUS│RESULT│     │
         │      │      │      │      │      │      │      │      │     │
UART:    │══════│══════│══════│══════│══════│══════│══════│══════│═════│
         │ Cmd  │ Bulk │ Bulk │ Start│ Poll │ Poll │ Poll │ Read │ Done│
         │Pkts  │Xfer  │Xfer  │ Pkt  │ Pkt  │ Pkt  │ Pkt  │ Pkt  │     │
         │      │      │      │      │      │      │      │      │     │
Hardware:│ CSR  │Buffer│Buffer│Compute────────────▶│Status│Result│     │
         │Write │Load  │Load  │ Start │     │     │ Rdy  │ Send │     │
         │      │      │      │      │     │     │      │      │     │

Throughput Analysis:
▪ UART Bandwidth: 115.2 kbps = 14.4 KB/s
▪ CSR Config: ~12 registers × 4B = 48B → ~3.3ms
▪ 2×2 Tile Data: 2×2×2 (A) + 2×2×2 (B) = 16B → ~1.1ms  
▪ Computation: 2×2×2 MACs = 8 ops @ 100MHz → 80ns
▪ Result Transfer: 2×2×4B = 16B → ~1.1ms
▪ Total Latency: ~5.5ms per tile (dominated by UART, not compute)

Optimization Opportunities:
▪ Higher UART baud rate (921.6k, 3M bps)
▪ Batch CSR writes in single packet
▪ Compressed data formats
▪ Tile size optimization (larger tiles = better compute/communication ratio)
```