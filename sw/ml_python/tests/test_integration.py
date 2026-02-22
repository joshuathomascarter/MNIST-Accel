#!/usr/bin/env python3
"""
test_integration.py - Integration tests for ACCEL-v1 AXI host tiler

Tests end-to-end functionality:
  - CSR register packing/serialization round-trips
  - GEMMConfig validation and edge cases
  - Tiled matrix multiplication correctness
  - AXI host tiler initialization and configuration
  - Matrix I/O save/load
"""

import unittest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host_axi.csr_map import (
    Config,
    pack_u32,
    unpack_u32,
    pack_f32,
    unpack_f32,
    pack_CTRL,
    unpack_CTRL,
    pack_DIMS,
    unpack_DIMS,
    pack_TILES,
    unpack_TILES,
    pack_INDEX,
    unpack_INDEX,
    pack_BUFF,
    unpack_BUFF,
    pack_SCALE,
    unpack_SCALE,
    pack_STATUS,
    unpack_STATUS,
    pack_DMA_CFG,
    unpack_DMA_CFG,
    to_writes,
    make_ctrl_start,
    make_ctrl_abort,
    CTRL,
    STATUS,
    STS_BUSY,
    STS_DONE_TILE,
    DIMS_M,
    DIMS_N,
    DIMS_K,
    TILES_Tm,
    TILES_Tn,
    TILES_Tk,
    CTRL_START,
    CTRL_ABORT,
    CTRL_IRQEN,
    WR_A,
    WR_B,
    BSR_CONFIG,
    BSR_CONFIG_MODE_BSR,
    BSR_CONFIG_MODE_DENSE,
)

from host_axi.run_gemm_axi import GEMMConfig, HostAXITiler


# ---------------------------------------------------------------------------
# 1. CSR Pack / Unpack
# ---------------------------------------------------------------------------
class TestCSRPackUnpack(unittest.TestCase):
    """Round-trip tests for every CSR register helper."""

    def test_u32_round_trip(self):
        for val in [0, 1, 0xDEADBEEF, 0xFFFFFFFF]:
            self.assertEqual(unpack_u32(pack_u32(val)), val)

    def test_f32_round_trip(self):
        for val in [0.0, 1.0, -3.14, 1e6]:
            self.assertAlmostEqual(unpack_f32(pack_f32(val)), val, places=2)

    def test_ctrl(self):
        for s, a, i in [(1, 0, 1), (0, 1, 0), (1, 1, 1), (0, 0, 0)]:
            self.assertEqual(unpack_CTRL(pack_CTRL(s, a, i)), (s, a, i))

    def test_dims(self):
        self.assertEqual(unpack_DIMS(pack_DIMS(7, 11, 22)), (7, 11, 22))

    def test_tiles(self):
        self.assertEqual(unpack_TILES(pack_TILES(1, 2, 3)), (1, 2, 3))

    def test_index(self):
        self.assertEqual(unpack_INDEX(pack_INDEX(5, 9)), (5, 9))

    def test_buff(self):
        self.assertEqual(unpack_BUFF(pack_BUFF(1, 0, 1, 1)), (1, 0, 1, 1))
        self.assertEqual(unpack_BUFF(pack_BUFF(0, 1, 0, 0)), (0, 1, 0, 0))

    def test_scale(self):
        self.assertEqual(unpack_SCALE(pack_SCALE(55, 77)), (55, 77))

    def test_status(self):
        for b, d, e in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
            self.assertEqual(unpack_STATUS(pack_STATUS(b, d, e)), (b, d, e))

    def test_dma_cfg(self):
        self.assertEqual(unpack_DMA_CFG(pack_DMA_CFG(256, 1)), (256, 1))


# ---------------------------------------------------------------------------
# 2. Config Serialization
# ---------------------------------------------------------------------------
class TestConfigSerialization(unittest.TestCase):
    """Config.to_bytes() / Config.from_bytes() round-trip."""

    def test_basic_round_trip(self):
        cfg = Config(M=7, N=11, K=22, Tm=1, Tn=2, Tk=3,
                     m_idx=5, n_idx=9, k_idx=13, Sa=0.5, Sw=2.0,
                     wrA=1, wrB=0)
        reg_img = cfg.to_bytes()
        cfg2 = Config.from_bytes(reg_img)
        self.assertEqual(cfg2.M, 7)
        self.assertEqual(cfg2.N, 11)
        self.assertEqual(cfg2.K, 22)
        self.assertEqual(cfg2.Tm, 1)
        self.assertEqual(cfg2.Tn, 2)
        self.assertEqual(cfg2.Tk, 3)
        self.assertEqual(cfg2.m_idx, 5)
        self.assertEqual(cfg2.n_idx, 9)
        self.assertAlmostEqual(cfg2.Sa, 0.5, places=5)
        self.assertAlmostEqual(cfg2.Sw, 2.0, places=5)

    def test_dims_field_layout(self):
        cfg = Config(M=14, N=14, K=14, Tm=14, Tn=14, Tk=14)
        reg_img = cfg.to_bytes()
        self.assertEqual(reg_img[DIMS_M:DIMS_M + 4], pack_u32(14))
        self.assertEqual(reg_img[DIMS_N:DIMS_N + 4], pack_u32(14))
        self.assertEqual(reg_img[DIMS_K:DIMS_K + 4], pack_u32(14))

    def test_to_writes_list(self):
        cfg = Config(M=8, N=8, K=8, Tm=4, Tn=4, Tk=4,
                     Sa=1.0, Sw=1.0, wrA=1, wrB=0)
        writes = to_writes(cfg)
        self.assertIsInstance(writes, list)
        self.assertTrue(len(writes) > 0)
        addrs = [addr for addr, _ in writes]
        self.assertIn(DIMS_M, addrs)
        self.assertIn(TILES_Tm, addrs)

    def test_ctrl_start(self):
        """make_ctrl_start produces a START pulse with optional IRQ bit."""
        raw = make_ctrl_start(irq_en=True)
        word = unpack_u32(raw)
        self.assertTrue(word & CTRL_START)
        self.assertTrue(word & CTRL_IRQEN)

    def test_ctrl_abort(self):
        raw = make_ctrl_abort()
        word = unpack_u32(raw)
        self.assertTrue(word & CTRL_ABORT)


# ---------------------------------------------------------------------------
# 3. GEMMConfig Validation
# ---------------------------------------------------------------------------
class TestGEMMConfig(unittest.TestCase):
    """Validate GEMMConfig constructor checks."""

    def test_valid_config(self):
        cfg = GEMMConfig(M=8, N=8, K=8, Tm=2, Tn=2, Tk=2)
        self.assertEqual(cfg.M, 8)
        self.assertEqual(cfg.N, 8)
        self.assertEqual(cfg.K, 8)

    def test_negative_dims(self):
        with self.assertRaises(ValueError):
            GEMMConfig(M=0, N=8, K=8, Tm=2, Tn=2, Tk=2)
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=-1, K=8, Tm=2, Tn=2, Tk=2)

    def test_negative_tiles(self):
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=8, K=8, Tm=0, Tn=2, Tk=2)

    def test_divisibility(self):
        with self.assertRaises(ValueError):
            GEMMConfig(M=9, N=8, K=8, Tm=2, Tn=2, Tk=2)
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=9, K=8, Tm=2, Tn=2, Tk=2)


# ---------------------------------------------------------------------------
# 4. Tiled GEMM Correctness
# ---------------------------------------------------------------------------
class TestTiledGEMM(unittest.TestCase):
    """Verify tiled matrix multiply matches numpy reference."""

    @staticmethod
    def _tiled_gemm(A, B, Tm, Tn, Tk):
        M, K = A.shape
        _, N = B.shape
        C = np.zeros((M, N), dtype=np.int32)
        for m in range(0, M, Tm):
            for n in range(0, N, Tn):
                for k in range(0, K, Tk):
                    a_tile = A[m:m + Tm, k:k + Tk].astype(np.int32)
                    b_tile = B[k:k + Tk, n:n + Tn].astype(np.int32)
                    C[m:m + Tm, n:n + Tn] += a_tile @ b_tile
        return C

    def test_2x2_identity(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        C = self._tiled_gemm(A, B, 2, 2, 2)
        expected = np.array([[19, 22], [43, 50]], dtype=np.int32)
        np.testing.assert_array_equal(C, expected)

    def test_8x8_random(self):
        rng = np.random.default_rng(42)
        A = rng.integers(-16, 16, size=(8, 8), dtype=np.int8)
        B = rng.integers(-16, 16, size=(8, 8), dtype=np.int8)
        C = self._tiled_gemm(A, B, 4, 4, 4)
        ref = A.astype(np.int32) @ B.astype(np.int32)
        np.testing.assert_array_equal(C, ref)

    def test_non_square(self):
        rng = np.random.default_rng(99)
        M, N, K = 12, 8, 16
        A = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        B = rng.integers(-128, 127, size=(K, N), dtype=np.int8)
        C = self._tiled_gemm(A, B, 4, 4, 4)
        ref = A.astype(np.int32) @ B.astype(np.int32)
        np.testing.assert_array_equal(C, ref)

    def test_14x14_systolic_size(self):
        """14×14 matches the physical array dimension."""
        rng = np.random.default_rng(7)
        A = rng.integers(-128, 127, size=(14, 14), dtype=np.int8)
        B = rng.integers(-128, 127, size=(14, 14), dtype=np.int8)
        C = self._tiled_gemm(A, B, 14, 14, 14)
        ref = A.astype(np.int32) @ B.astype(np.int32)
        np.testing.assert_array_equal(C, ref)

    def test_accumulation_across_k(self):
        """Partial products along K must sum correctly."""
        rng = np.random.default_rng(55)
        A = rng.integers(-10, 10, size=(4, 8), dtype=np.int8)
        B = rng.integers(-10, 10, size=(8, 4), dtype=np.int8)
        C = self._tiled_gemm(A, B, 4, 4, 2)
        ref = A.astype(np.int32) @ B.astype(np.int32)
        np.testing.assert_array_equal(C, ref)


# ---------------------------------------------------------------------------
# 5. HostAXITiler (simulator mode)
# ---------------------------------------------------------------------------
class TestHostAXITiler(unittest.TestCase):
    """Test AXI host tiler in pure-software simulator mode."""

    def test_create_and_close(self):
        tiler = HostAXITiler(use_simulator=True, verbose=False)
        tiler.close()

    def test_context_manager(self):
        with HostAXITiler(use_simulator=True, verbose=False) as t:
            pass

    def test_configure_tile(self):
        with HostAXITiler(use_simulator=True, verbose=False) as t:
            cfg = GEMMConfig(M=8, N=8, K=8, Tm=4, Tn=4, Tk=4)
            ok = t.configure_accelerator(cfg, m_idx=0, n_idx=0, k_idx=0)
            self.assertTrue(ok)


# ---------------------------------------------------------------------------
# 6. Matrix I/O
# ---------------------------------------------------------------------------
class TestMatrixIO(unittest.TestCase):
    """Save / load matrices through numpy."""

    def test_npz_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            rng = np.random.default_rng(456)
            A = rng.integers(-128, 127, size=(4, 4), dtype=np.int8)
            B = rng.integers(-128, 127, size=(4, 4), dtype=np.int8)
            np.savez(path, A=A, B=B)
            data = np.load(path)
            np.testing.assert_array_equal(A, data["A"])
            np.testing.assert_array_equal(B, data["B"])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 7. BSR Config Bits
# ---------------------------------------------------------------------------
class TestBSRConfig(unittest.TestCase):
    """Verify BSR scheduler-select constants."""

    def test_mode_bits(self):
        self.assertEqual(BSR_CONFIG_MODE_BSR, 0)
        self.assertEqual(BSR_CONFIG_MODE_DENSE, 1)

    def test_bsr_config_address(self):
        self.assertEqual(BSR_CONFIG, 0xC0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
