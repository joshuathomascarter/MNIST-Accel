/*
 * hal_accel.h — Hardware Abstraction Layer for Accelerator Tile Array
 * ====================================================================
 * CSR address map per tile (base = ACCEL_BASE + tile_id * 0x1000):
 *   0x00  CMD_OPCODE  (write triggers command; arg regs must be set first)
 *   0x04  CMD_ARG0    (DMA source addr / generic arg)
 *   0x08  CMD_ARG1    (DMA length  / generic arg)
 *   0x0C  CMD_ARG2    (scratchpad offset / generic arg)
 *   0x10  STATUS      (read: bit 0 = busy, bit 1 = done)
 *
 * Tile-array / NoC metadata CSR window (base = ACCEL_BASE, tile selector ignored):
 *   0x80  NOC_META_ROUTER     [3:0]  router/node id to program
 *   0x84  NOC_META_REDUCE_ID  [7:0]  reduction group id
 *   0x88  NOC_META_TARGET     [3:0]  local subtree completion target
 *   0x8C  NOC_META_CTRL       [0] enable, [1] apply (W1P)
 *   0x90  NOC_META_STATUS     latched programming state snapshot
 */

#ifndef HAL_ACCEL_H
#define HAL_ACCEL_H

#include <stdint.h>
#include <stdbool.h>

/* ---- Base address (from hal.h) ---------------------------------------- */
#ifndef ACCEL_BASE
#define ACCEL_BASE   0x30000000u
#endif

/* ---- Mesh geometry ---------------------------------------------------- */
#define ACCEL_MESH_ROWS   4
#define ACCEL_MESH_COLS   4
#define ACCEL_NUM_TILES   (ACCEL_MESH_ROWS * ACCEL_MESH_COLS)

/* ---- Per-tile CSR offsets --------------------------------------------- */
#define ACCEL_CSR_CMD      0x00u
#define ACCEL_CSR_ARG0     0x04u
#define ACCEL_CSR_ARG1     0x08u
#define ACCEL_CSR_ARG2     0x0Cu
#define ACCEL_CSR_STATUS   0x10u

/* ---- Tile address helper ---------------------------------------------- */
#define ACCEL_TILE_BASE(tile)  (ACCEL_BASE + ((uint32_t)(tile) << 12))
#define ACCEL_TILE_REG(tile, off) \
    (*(volatile uint32_t *)(uintptr_t)(ACCEL_TILE_BASE(tile) + (off)))

/* ---- Tile-array / NoC metadata registers ------------------------------ */
#define ACCEL_NOC_META_ROUTER     (ACCEL_BASE + 0x80u)
#define ACCEL_NOC_META_REDUCE_ID  (ACCEL_BASE + 0x84u)
#define ACCEL_NOC_META_TARGET     (ACCEL_BASE + 0x88u)
#define ACCEL_NOC_META_CTRL       (ACCEL_BASE + 0x8Cu)
#define ACCEL_NOC_META_STATUS     (ACCEL_BASE + 0x90u)

/* ---- Command opcodes (must match tile_controller.sv) ------------------ */
#define OP_NOP       0x00u
#define OP_LOAD      0x01u   /* DMA read  → scratchpad              */
#define OP_STORE     0x02u   /* scratchpad → DMA write              */
#define OP_COMPUTE   0x03u   /* Run systolic array                  */
#define OP_BARRIER   0x04u   /* Synchronise with other tiles        */
#define OP_SPARSE    0x05u   /* Set sparse-hint flag                */

/* ---- Status bits ------------------------------------------------------ */
#define ACCEL_STATUS_BUSY  (1u << 0)
#define ACCEL_STATUS_DONE  (1u << 1)

/* ====================================================================== */
/*  Low-level helpers                                                     */
/* ====================================================================== */

/* Write arg registers, then issue command (order matters: args before cmd) */
static inline void accel_cmd(uint32_t tile,
                             uint32_t opcode,
                             uint32_t arg0,
                             uint32_t arg1,
                             uint32_t arg2)
{
    ACCEL_TILE_REG(tile, ACCEL_CSR_ARG0) = arg0;
    ACCEL_TILE_REG(tile, ACCEL_CSR_ARG1) = arg1;
    ACCEL_TILE_REG(tile, ACCEL_CSR_ARG2) = arg2;
    ACCEL_TILE_REG(tile, ACCEL_CSR_CMD)  = opcode;
}

/* Read status register */
static inline uint32_t accel_status(uint32_t tile)
{
    return ACCEL_TILE_REG(tile, ACCEL_CSR_STATUS);
}

/* Poll until tile is idle (busy bit cleared) */
static inline void accel_wait_idle(uint32_t tile)
{
    while (accel_status(tile) & ACCEL_STATUS_BUSY)
        ;
}

/* Issue OP_LOAD: DMA `len` 64-bit words from `src_addr` into scratchpad at offset `sp_off` */
static inline void accel_load(uint32_t tile,
                              uint32_t src_addr,
                              uint16_t len,
                              uint32_t sp_off)
{
    accel_wait_idle(tile);
    accel_cmd(tile, OP_LOAD, src_addr, (uint32_t)len, sp_off);
}

/* Issue OP_STORE: DMA `len` words from scratchpad at offset `sp_off` to `dst_addr` */
static inline void accel_store(uint32_t tile,
                               uint32_t dst_addr,
                               uint16_t len,
                               uint32_t sp_off)
{
    accel_wait_idle(tile);
    accel_cmd(tile, OP_STORE, dst_addr, (uint32_t)len, sp_off);
}

/* Issue OP_COMPUTE: start systolic array, poll until done */
static inline void accel_compute_at(uint32_t tile,
                                    uint32_t act_sp_off,
                                    uint32_t wgt_sp_off,
                                    uint32_t out_sp_off)
{
    accel_wait_idle(tile);
    accel_cmd(tile, OP_COMPUTE, act_sp_off, wgt_sp_off, out_sp_off);
}

static inline void accel_compute(uint32_t tile)
{
    accel_compute_at(tile, 0, 0, 0);
}

/* Issue OP_BARRIER: wait for all tiles to reach barrier */
static inline void accel_barrier(uint32_t tile)
{
    accel_wait_idle(tile);
    accel_cmd(tile, OP_BARRIER, 0, 0, 0);
}

/* Set sparse-hint for a tile (feeds NI VC allocator) */
static inline void accel_set_sparse(uint32_t tile, bool enable)
{
    accel_wait_idle(tile);
    accel_cmd(tile, OP_SPARSE, enable ? 1u : 0u, 0, 0);
}

/* Program one router-local subtree target for one reduce_id.
 * enable=true installs the explicit target; enable=false clears it.
 */
static inline void accel_noc_program_reduce_target(uint32_t router,
                                                   uint32_t reduce_id,
                                                   uint32_t target,
                                                   bool enable)
{
    REG32(ACCEL_NOC_META_ROUTER)    = router & 0xFu;
    REG32(ACCEL_NOC_META_REDUCE_ID) = reduce_id & 0xFFu;
    REG32(ACCEL_NOC_META_TARGET)    = target & 0xFu;
    REG32(ACCEL_NOC_META_CTRL)      = (enable ? 1u : 0u) | (1u << 1);
}

static inline void accel_noc_clear_reduce_target(uint32_t router,
                                                 uint32_t reduce_id)
{
    accel_noc_program_reduce_target(router, reduce_id, 0u, false);
}

/* Wait until ALL tiles are idle */
static inline void accel_wait_all_idle(void)
{
    for (uint32_t t = 0; t < ACCEL_NUM_TILES; t++)
        accel_wait_idle(t);
}

/* Issue barrier on all tiles simultaneously, then wait for all to complete */
static inline void accel_barrier_all(void)
{
    for (uint32_t t = 0; t < ACCEL_NUM_TILES; t++)
        accel_cmd(t, OP_BARRIER, 0, 0, 0);
    accel_wait_all_idle();
}

#endif /* HAL_ACCEL_H */
