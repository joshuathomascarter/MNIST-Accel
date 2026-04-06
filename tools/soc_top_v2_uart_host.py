#!/usr/bin/env python3
"""UART host for soc_top_v2 block-level preload, compute, and readback."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import serial

CMD_PING = 0x00
CMD_MEM_WRITE_WORDS = 0x01
CMD_MEM_READ_WORDS = 0x02
CMD_TILE_LOAD = 0x10
CMD_TILE_COMPUTE = 0x11
CMD_TILE_STORE = 0x12
CMD_TILE_STATUS = 0x13
CMD_TILE_BARRIER_ALL = 0x14
CMD_ARGMAX = 0x20

RESP_OK = 0x79
RESP_ERR = 0x1F

DRAM_BASE = 0x4000_0000
DRAM_BASE_UC = 0x6000_0000

ACT_DRAM_OFFSET = 0x0000
WGT_DRAM_OFFSET = 0x1000
OUT_DRAM_OFFSET = 0x2000

SP_ACT_BASE = 0x000
SP_WGT_BASE = 0x100
SP_OUT_BASE = 0x200

ACT_WORDS = 64
WGT_WORDS = 64
OUT_WORDS = 256


class SocTopV2UartHost:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0) -> None:
        self.serial = serial.Serial(port=port, baudrate=baud, timeout=timeout)

    def close(self) -> None:
        self.serial.close()

    def __enter__(self) -> "SocTopV2UartHost":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _write_u8(self, value: int) -> None:
        self.serial.write(bytes((value & 0xFF,)))

    def _write_u32(self, value: int) -> None:
        self.serial.write(struct.pack("<I", value & 0xFFFF_FFFF))

    def _read_exact(self, count: int) -> bytes:
        data = self.serial.read(count)
        if len(data) != count:
            raise TimeoutError(f"expected {count} bytes, received {len(data)}")
        return data

    def _read_u32(self) -> int:
        return struct.unpack("<I", self._read_exact(4))[0]

    def _expect_ok(self) -> None:
        status = self._read_exact(1)[0]
        if status == RESP_OK:
            return
        if status == RESP_ERR:
            raise RuntimeError(f"device error {self._read_u32()}")
        raise RuntimeError(f"unexpected device status 0x{status:02x}")

    def ping(self) -> int:
        self._write_u8(CMD_PING)
        self._expect_ok()
        return self._read_u32()

    def mem_write_words(self, addr: int, words: Iterable[int]) -> None:
        word_list = [int(word) & 0xFFFF_FFFF for word in words]
        self._write_u8(CMD_MEM_WRITE_WORDS)
        self._write_u32(addr)
        self._write_u32(len(word_list))
        for word in word_list:
            self._write_u32(word)
        self._expect_ok()

    def mem_read_words(self, addr: int, word_count: int) -> list[int]:
        self._write_u8(CMD_MEM_READ_WORDS)
        self._write_u32(addr)
        self._write_u32(word_count)
        self._expect_ok()
        return [self._read_u32() for _ in range(word_count)]

    def tile_load(self, tile: int, dram_addr: int, word_count: int, sp_off: int) -> None:
        self._write_u8(CMD_TILE_LOAD)
        self._write_u32(tile)
        self._write_u32(dram_addr)
        self._write_u32(word_count)
        self._write_u32(sp_off)
        self._expect_ok()

    def tile_compute(self, tile: int, act_sp_off: int, wgt_sp_off: int, out_sp_off: int) -> None:
        self._write_u8(CMD_TILE_COMPUTE)
        self._write_u32(tile)
        self._write_u32(act_sp_off)
        self._write_u32(wgt_sp_off)
        self._write_u32(out_sp_off)
        self._expect_ok()

    def tile_store(self, tile: int, dram_addr: int, word_count: int, sp_off: int) -> None:
        self._write_u8(CMD_TILE_STORE)
        self._write_u32(tile)
        self._write_u32(dram_addr)
        self._write_u32(word_count)
        self._write_u32(sp_off)
        self._expect_ok()

    def tile_status(self, tile: int) -> int:
        self._write_u8(CMD_TILE_STATUS)
        self._write_u32(tile)
        self._expect_ok()
        return self._read_u32()

    def argmax(self, addr: int, count: int) -> tuple[int, int]:
        self._write_u8(CMD_ARGMAX)
        self._write_u32(addr)
        self._write_u32(count)
        self._expect_ok()
        return self._read_u32(), self._read_u32()


def _load_first_block(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    flat = np.asarray(data, dtype=np.int32).reshape(-1)
    block = np.zeros(16 * 16, dtype=np.int8)
    count = min(flat.size, block.size)
    block[:count] = np.clip(flat[:count], -128, 127).astype(np.int8)
    return block.reshape(16, 16)


def _default_act_block() -> np.ndarray:
    values = np.arange(16 * 16, dtype=np.int16) % 13
    return values.astype(np.int8).reshape(16, 16)


def _default_weight_block() -> np.ndarray:
    block = np.zeros((16, 16), dtype=np.int8)
    np.fill_diagonal(block, 1)
    return block


def _pack_int8_block(block: np.ndarray) -> list[int]:
    block_i8 = np.asarray(block, dtype=np.int8).reshape(16, 16)
    words: list[int] = []
    for row in block_i8:
        row_bytes = row.astype(np.int8).view(np.uint8)
        for idx in range(0, 16, 4):
            words.append(
                int(row_bytes[idx])
                | (int(row_bytes[idx + 1]) << 8)
                | (int(row_bytes[idx + 2]) << 16)
                | (int(row_bytes[idx + 3]) << 24)
            )
    return words


def _unpack_int32_words(words: Iterable[int]) -> np.ndarray:
    values = np.fromiter((np.int32(word).item() for word in words), dtype=np.int32, count=OUT_WORDS)
    return values.reshape(16, 16)


def run_block_demo(host: SocTopV2UartHost, act_block: np.ndarray, weight_block: np.ndarray, tile: int) -> np.ndarray:
    act_words = _pack_int8_block(act_block)
    weight_words = _pack_int8_block(weight_block)

    host.mem_write_words(DRAM_BASE_UC + ACT_DRAM_OFFSET, act_words)
    host.mem_write_words(DRAM_BASE_UC + WGT_DRAM_OFFSET, weight_words)

    host.tile_load(tile, DRAM_BASE + ACT_DRAM_OFFSET, ACT_WORDS, SP_ACT_BASE)
    host.tile_load(tile, DRAM_BASE + WGT_DRAM_OFFSET, WGT_WORDS, SP_WGT_BASE)
    host.tile_compute(tile, SP_ACT_BASE, SP_WGT_BASE, SP_OUT_BASE)
    host.tile_store(tile, DRAM_BASE + OUT_DRAM_OFFSET, OUT_WORDS, SP_OUT_BASE)

    return _unpack_int32_words(host.mem_read_words(DRAM_BASE_UC + OUT_DRAM_OFFSET, OUT_WORDS))


def run_cli(default_flow: str) -> int:
    parser = argparse.ArgumentParser(description=f"soc_top_v2 {default_flow} UART block demo")
    parser.add_argument("--port", required=True, help="Serial port connected to soc_top_v2 UART")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate")
    parser.add_argument("--tile", type=int, default=0, help="Tile ID to run")
    parser.add_argument("--act-npy", type=Path, help="Optional .npy file to source the first 16x16 activation block")
    parser.add_argument("--weight-npy", type=Path, help="Optional .npy file to source the first 16x16 weight block")
    parser.add_argument("--print-matrix", action="store_true", help="Print the full 16x16 int32 output block")
    args = parser.parse_args()

    act_block = _load_first_block(args.act_npy) if args.act_npy else _default_act_block()
    weight_block = _load_first_block(args.weight_npy) if args.weight_npy else _default_weight_block()

    with SocTopV2UartHost(args.port, baud=args.baud) as host:
        magic = host.ping()
        if magic != 0x53545632:
            raise RuntimeError(f"unexpected ping response 0x{magic:08x}")

        output = run_block_demo(host, act_block, weight_block, tile=args.tile)
        argmax_idx, argmax_val = host.argmax(DRAM_BASE_UC + OUT_DRAM_OFFSET, OUT_WORDS)

    print(f"flow={default_flow} tile={args.tile} argmax={argmax_idx} value={argmax_val}")
    if args.print_matrix:
        np.set_printoptions(linewidth=120)
        print(output)
    else:
        print(output[:4, :4])
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli("generic"))