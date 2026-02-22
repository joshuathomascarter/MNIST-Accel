#!/usr/bin/env python3
"""
accel.py — PYNQ Driver for ACCEL-v1 Sparse Neural Network Accelerator
======================================================================

This is the main driver class for controlling the ACCEL-v1 hardware accelerator
on Zynq-based boards using the PYNQ framework.

Features:
  - CSR configuration via AXI-Lite
  - DMA transfers for weights and activations
  - Sparse BSR weight loading
  - Full matrix multiply execution
  - Performance monitoring

Usage:
    from accel import AccelDriver
    
    accel = AccelDriver(overlay)
    accel.load_sparse_weights(row_ptr, col_idx, weights)
    accel.load_activations(activations)
    result = accel.run_inference()

Author: Joshua Carter
Date: December 2024
"""

import numpy as np
from typing import Optional, Tuple, List
import time

# Try to import PYNQ (will fail on non-Zynq systems)
try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[accel.py] Warning: PYNQ not available, running in simulation mode")


class CSRMap:
    """CSR Address Map for ACCEL-v1 (matches csr.sv)"""
    
    # Control registers
    CTRL        = 0x00  # [0]=start, [1]=abort, [2]=irq_en
    DIMS_M      = 0x04  # Matrix dimension M
    DIMS_N      = 0x08  # Matrix dimension N
    DIMS_K      = 0x0C  # Matrix dimension K
    TILES_Tm    = 0x10  # Tile size M
    TILES_Tn    = 0x14  # Tile size N
    TILES_Tk    = 0x18  # Tile size K
    INDEX_m     = 0x1C  # Current M index
    INDEX_n     = 0x20  # Current N index
    INDEX_k     = 0x24  # Current K index
    BUFF        = 0x28  # Buffer control
    SCALE_Sa    = 0x2C  # Activation scale factor
    SCALE_Sw    = 0x30  # Weight scale factor
    
    # Status register
    STATUS      = 0x3C  # [0]=busy, [1]=done_tile, [9]=error
    
    # Performance counters (Read-Only)
    PERF_TOTAL      = 0x40  # Total cycles from start to done
    PERF_ACTIVE     = 0x44  # Cycles where busy was high
    PERF_IDLE       = 0x48  # Cycles where busy was low
    PERF_DMA_BYTES  = 0x4C  # Total DMA bytes transferred
    PERF_BLOCKS     = 0x50  # Non-zero BSR blocks computed
    PERF_STALL      = 0x54  # Scheduler busy, PEs idle (stall cycles)
    
    # Results (first 4 output accumulators)
    RESULT_0    = 0x80
    RESULT_1    = 0x84
    RESULT_2    = 0x88
    RESULT_3    = 0x8C
    
    # DMA registers
    DMA_SRC_ADDR     = 0x90  # BSR DMA source address
    DMA_DST_ADDR     = 0x94  # (unused)
    DMA_XFER_LEN     = 0x98  # BSR DMA transfer length
    DMA_CTRL         = 0x9C  # [0]=start, [1]=busy, [2]=done
    ACT_DMA_SRC_ADDR = 0xA0  # Activation DMA source address
    ACT_DMA_LEN      = 0xA4  # Activation DMA length
    ACT_DMA_CTRL     = 0xA8  # Activation DMA control
    
    # DMA bytes transferred (Read-Only)
    DMA_BYTES_XFERRED = 0xB8  # Total bytes transferred by DMA

    # BSR / Hybrid Scheduler registers (0xC0 - 0xDF)
    # The accelerator has two schedulers sharing the same 14×14 systolic array:
    #   - BSR Scheduler: For sparse layers with BSR weights
    #   - Dense Scheduler: For FC layers (100% dense)
    # BSR_CONFIG[0] selects which scheduler: 0=BSR, 1=Dense
    BSR_CONFIG       = 0xC0  # Scheduler mode & BSR config
    BSR_NUM_BLOCKS   = 0xC4  # Number of non-zero BSR blocks
    BSR_BLOCK_ROWS   = 0xC8  # Block grid rows
    BSR_BLOCK_COLS   = 0xCC  # Block grid columns
    BSR_STATUS       = 0xD0  # BSR engine status
    BSR_ERROR_CODE   = 0xD4  # BSR error detail code (RO)
    BSR_PTR_ADDR     = 0xD8  # row_ptr array address
    BSR_IDX_ADDR     = 0xDC  # col_idx array address
    
    # BSR_CONFIG bits
    SCHED_MODE_BSR   = 0      # Use BSR sparse scheduler
    SCHED_MODE_DENSE = 1 << 0 # Use Dense GEMM scheduler


class AccelDriver:
    """
    PYNQ Driver for ACCEL-v1 Sparse Neural Network Accelerator.
    
    This driver provides a high-level interface for:
    - Loading sparse BSR weights
    - Loading dense activations
    - Running matrix multiply inference
    - Reading results and performance counters
    """
    
    # Hardware constants
    BLOCK_SIZE = 14  # 14x14 systolic array (PYNQ-Z2)
    DATA_WIDTH = 8   # INT8
    ACC_WIDTH = 32   # INT32 accumulators
    
    def __init__(self, overlay=None, csr_base: int = 0x43C00000, 
                 dma_base: int = 0x40000000, simulation: bool = False):
        """
        Initialize the accelerator driver.
        
        Args:
            overlay: PYNQ Overlay object (None for simulation)
            csr_base: Base address of AXI-Lite CSR slave
            dma_base: Base address for DMA buffers
            simulation: Run in simulation mode (no hardware)
        """
        self.csr_base = csr_base
        self.dma_base = dma_base
        self.simulation = simulation or not PYNQ_AVAILABLE
        
        if not self.simulation and overlay is not None:
            # Real hardware mode
            self.overlay = overlay
            self.csr = overlay.axi_lite_0
            
            # Allocate DMA buffers
            self.weight_buffer = None
            self.act_buffer = None
            self.result_buffer = None
        else:
            # Simulation mode
            self.overlay = None
            self.csr = SimulatedCSR()
            self._sim_memory = {}
        
        # State tracking
        self.M = 0
        self.N = 0
        self.K = 0
        self.weights_loaded = False
        self.activations_loaded = False
        
    def configure_dimensions(self, M: int, N: int, K: int, 
                            Tm: int = 14, Tn: int = 14, Tk: int = 14):
        """
        Configure matrix dimensions.
        
        Args:
            M: Output rows (activation rows)
            N: Output columns (weight columns)
            K: Reduction dimension (shared)
            Tm, Tn, Tk: Tile sizes (default 14 for 14x14 array)
        """
        self.M = M
        self.N = N
        self.K = K
        
        self._csr_write(CSRMap.DIMS_M, M)
        self._csr_write(CSRMap.DIMS_N, N)
        self._csr_write(CSRMap.DIMS_K, K)
        self._csr_write(CSRMap.TILES_Tm, Tm)
        self._csr_write(CSRMap.TILES_Tn, Tn)
        self._csr_write(CSRMap.TILES_Tk, Tk)
        
    def load_sparse_weights(self, row_ptr: np.ndarray, col_idx: np.ndarray, 
                           weights: np.ndarray, block_size: int = 14) -> int:
        """
        Load sparse weights in BSR format.
        
        BSR (Block Sparse Row) format:
        - row_ptr: Array of pointers to start of each row's blocks
        - col_idx: Column indices for each non-zero block
        - weights: Dense blocks of shape (num_blocks, block_size, block_size)
        
        Args:
            row_ptr: Row pointer array (num_block_rows + 1,)
            col_idx: Column index array (num_blocks,)
            weights: Weight blocks (num_blocks, block_size, block_size) INT8
            block_size: Block dimension (must match hardware, default 14)
            
        Returns:
            Total bytes transferred
        """
        assert block_size == self.BLOCK_SIZE, f"Block size must be {self.BLOCK_SIZE}"
        assert weights.dtype == np.int8, "Weights must be INT8"
        
        num_blocks = len(col_idx)
        
        # Pack BSR data into contiguous buffer
        # Layout: [row_ptr (4B each)][col_idx (2B each)][weights (block_size^2 each)]
        
        row_ptr_bytes = row_ptr.astype(np.uint32).tobytes()
        col_idx_bytes = col_idx.astype(np.uint16).tobytes()
        weight_bytes = weights.astype(np.int8).tobytes()
        
        total_bytes = len(row_ptr_bytes) + len(col_idx_bytes) + len(weight_bytes)
        
        if self.simulation:
            # Simulation: store in memory dict
            self._sim_memory['bsr'] = row_ptr_bytes + col_idx_bytes + weight_bytes
            bsr_addr = self.dma_base
        else:
            # Real hardware: allocate and copy
            if self.weight_buffer is None or len(self.weight_buffer) < total_bytes:
                self.weight_buffer = allocate(shape=(total_bytes,), dtype=np.uint8)
            
            # Copy data
            buf = np.frombuffer(row_ptr_bytes + col_idx_bytes + weight_bytes, dtype=np.uint8)
            self.weight_buffer[:len(buf)] = buf
            self.weight_buffer.sync_to_device()
            bsr_addr = self.weight_buffer.device_address
        
        # Configure BSR DMA
        self._csr_write(CSRMap.DMA_SRC_ADDR, bsr_addr)
        self._csr_write(CSRMap.DMA_XFER_LEN, total_bytes)
        
        # Start DMA
        self._csr_write(CSRMap.DMA_CTRL, 0x1)
        
        # Wait for completion
        self._wait_dma_done(CSRMap.DMA_CTRL)
        
        self.weights_loaded = True
        return total_bytes
        
    def load_activations(self, activations: np.ndarray) -> int:
        """
        Load dense activation matrix.
        
        Args:
            activations: Activation matrix (M, K) INT8
            
        Returns:
            Total bytes transferred
        """
        assert activations.dtype == np.int8, "Activations must be INT8"
        assert activations.shape == (self.M, self.K), \
            f"Activation shape {activations.shape} doesn't match (M={self.M}, K={self.K})"
        
        act_bytes = activations.tobytes()
        total_bytes = len(act_bytes)
        
        if self.simulation:
            self._sim_memory['activations'] = act_bytes
            act_addr = self.dma_base + 0x100000  # Offset for activations
        else:
            if self.act_buffer is None or len(self.act_buffer) < total_bytes:
                self.act_buffer = allocate(shape=(total_bytes,), dtype=np.uint8)
            
            self.act_buffer[:total_bytes] = np.frombuffer(act_bytes, dtype=np.uint8)
            self.act_buffer.sync_to_device()
            act_addr = self.act_buffer.device_address
        
        # Configure activation DMA
        self._csr_write(CSRMap.ACT_DMA_SRC_ADDR, act_addr)
        self._csr_write(CSRMap.ACT_DMA_LEN, total_bytes)
        
        # Start DMA
        self._csr_write(CSRMap.ACT_DMA_CTRL, 0x1)
        
        # Wait for completion
        self._wait_dma_done(CSRMap.ACT_DMA_CTRL)
        
        self.activations_loaded = True
        return total_bytes
        
    def run_inference(self, timeout_ms: int = 1000) -> Tuple[bool, dict]:
        """
        Run sparse matrix multiply inference.
        
        Args:
            timeout_ms: Maximum wait time in milliseconds
            
        Returns:
            Tuple of (success, result_dict)
            result_dict contains:
              - 'cycles': Total cycles
              - 'active_cycles': Active cycles
              - 'utilization': Compute utilization percentage
              - 'result_sample': First 4 output values
        """
        assert self.weights_loaded, "Weights not loaded"
        assert self.activations_loaded, "Activations not loaded"
        
        # Clear status
        status = self._csr_read(CSRMap.STATUS)
        
        # Start computation
        self._csr_write(CSRMap.CTRL, 0x1)
        
        # Wait for done
        start_time = time.time()
        timeout_s = timeout_ms / 1000.0
        
        while True:
            status = self._csr_read(CSRMap.STATUS)
            done = (status >> 1) & 0x1
            error = (status >> 9) & 0x1
            
            if done or error:
                break
                
            if time.time() - start_time > timeout_s:
                return False, {'error': 'timeout'}
                
            time.sleep(0.001)  # 1ms poll interval
        
        # Read performance counters
        total_cycles = self._csr_read(CSRMap.PERF_TOTAL)
        active_cycles = self._csr_read(CSRMap.PERF_ACTIVE)
        
        utilization = 0.0
        if total_cycles > 0:
            utilization = (active_cycles / total_cycles) * 100.0
        
        # Read sample results
        results = [
            self._csr_read(CSRMap.RESULT_0),
            self._csr_read(CSRMap.RESULT_1),
            self._csr_read(CSRMap.RESULT_2),
            self._csr_read(CSRMap.RESULT_3),
        ]
        
        return not error, {
            'cycles': total_cycles,
            'active_cycles': active_cycles,
            'utilization': utilization,
            'result_sample': results,
            'error': error
        }
    
    def read_full_output(self, M: int, N: int) -> np.ndarray:
        """
        Read the full output matrix from DDR after compute completes.
        
        The output DMA (out_dma.sv) writes INT32 results back to DDR.
        This method reads them into a numpy array.
        
        Args:
            M: Output rows
            N: Output columns
            
        Returns:
            Output matrix (M, N) as INT32
        """
        total_elements = M * N
        total_bytes = total_elements * 4  # INT32 = 4 bytes
        
        if self.simulation:
            # Simulation: return zeros (no real compute)
            return np.zeros((M, N), dtype=np.int32)
        else:
            # Allocate result buffer if needed
            if self.result_buffer is None or len(self.result_buffer) < total_bytes:
                self.result_buffer = allocate(shape=(total_bytes,), dtype=np.uint8)
            
            # Sync from device (DDR → CPU)
            self.result_buffer.sync_from_device()
            
            # Interpret as INT32 and reshape
            raw = np.frombuffer(
                self.result_buffer[:total_bytes], dtype=np.int32
            )
            return raw.reshape(M, N).copy()

    def run_layer(self, layer_name: str,
                  row_ptr: np.ndarray, col_idx: np.ndarray,
                  weights: np.ndarray,
                  activations_2d: np.ndarray,
                  M: int, N: int, K: int,
                  Sa: float = 1.0, Sw: float = 1.0) -> np.ndarray:
        """
        Run a single GEMM layer end-to-end on the FPGA.
        
        This is the core dispatch function that:
          1. Configures matrix dimensions in CSRs
          2. Auto-selects BSR vs Dense scheduler
          3. DMA weights + activations into DDR
          4. Starts compute and waits for done
          5. Reads full output from DDR
        
        Args:
            layer_name: Human-readable name ("conv1", "fc2", etc.)
            row_ptr:    BSR row pointer array
            col_idx:    BSR column index array
            weights:    BSR weight blocks (num_blocks, 14, 14) INT8
            activations_2d: Activation matrix (M, K) INT8
            M: Output rows (num filters or neurons)
            N: Output columns (spatial positions or batch)
            K: Reduction dimension (input channels × kernel, or fan-in)
            Sa: Activation quantization scale
            Sw: Weight quantization scale
            
        Returns:
            Output matrix (M, N) as INT32
        """
        print(f"  [{layer_name}] GEMM: ({M}×{K}) × ({K}×{N}) → ({M}×{N})")
        
        # Step 1: Configure dimensions
        self.configure_dimensions(M, N, K)
        
        # Step 2: Set quantization scales
        self.set_scale_factors(Sa, Sw)
        
        # Step 3: Auto-select scheduler (BSR vs Dense)
        weight_2d = weights.reshape(-1, weights.shape[-1]) if weights.ndim > 2 else weights
        self.auto_select_scheduler(weight_2d, layer_name)
        
        # Step 4: Load weights via DMA
        wgt_bytes = self.load_sparse_weights(row_ptr, col_idx, weights)
        print(f"    Weights: {wgt_bytes} bytes → DDR")
        
        # Step 5: Load activations via DMA
        # Ensure shape matches (M, K) — transpose if needed for GEMM convention
        if activations_2d.shape != (M, K):
            if activations_2d.shape == (K, N):
                # Activations are (K, N) — the hardware expects (K, N) streamed
                # through columns, but load_activations wants (M, K).
                # For now, transpose to match CSR expectation.
                activations_2d = activations_2d.T
            else:
                raise ValueError(
                    f"Activation shape {activations_2d.shape} doesn't match "
                    f"expected (M={M}, K={K}) or (K={K}, N={N})"
                )
        act_bytes = self.load_activations(activations_2d)
        print(f"    Activations: {act_bytes} bytes → DDR")
        
        # Step 6: Run compute
        success, info = self.run_inference(timeout_ms=5000)
        if not success:
            raise RuntimeError(f"Layer {layer_name} failed: {info}")
        
        cycles = info.get('cycles', 0)
        util = info.get('utilization', 0)
        print(f"    Compute: {cycles} cycles, {util:.1f}% utilization")
        
        # Step 7: Read full output
        output = self.read_full_output(M, N)
        print(f"    Output: {output.shape} INT32")
        
        return output

    def get_performance_stats(self) -> dict:
        """
        Read detailed performance statistics.
        
        Returns:
            Dictionary with performance counters
        """
        return {
            'total_cycles': self._csr_read(CSRMap.PERF_TOTAL),
            'active_cycles': self._csr_read(CSRMap.PERF_ACTIVE),
            'idle_cycles': self._csr_read(CSRMap.PERF_IDLE),
            'dma_bytes': self._csr_read(CSRMap.PERF_DMA_BYTES),
            'blocks_done': self._csr_read(CSRMap.PERF_BLOCKS),
            'stall_cycles': self._csr_read(CSRMap.PERF_STALL),
        }
    
    def reset(self):
        """Reset the accelerator."""
        self._csr_write(CSRMap.CTRL, 0x2)  # Abort bit
        time.sleep(0.001)
        self._csr_write(CSRMap.CTRL, 0x0)
        
        self.weights_loaded = False
        self.activations_loaded = False
        
    def set_scale_factors(self, Sa: float, Sw: float):
        """
        Set quantization scale factors.
        
        Args:
            Sa: Activation scale (converted to Q16.16 fixed-point)
            Sw: Weight scale (converted to Q16.16 fixed-point)
        """
        # Convert to Q16.16 fixed-point
        Sa_fixed = int(Sa * 65536) & 0xFFFFFFFF
        Sw_fixed = int(Sw * 65536) & 0xFFFFFFFF
        
        self._csr_write(CSRMap.SCALE_Sa, Sa_fixed)
        self._csr_write(CSRMap.SCALE_Sw, Sw_fixed)
    
    def set_scheduler_mode(self, use_dense: bool):
        """
        Set scheduler mode for hybrid scheduler architecture.
        
        The accelerator has two schedulers sharing the same 14×14 systolic array:
          - BSR Scheduler: Optimized for Block Sparse Row format weights
          - Dense Scheduler: Traditional tiled GEMM for fully-connected layers
        
        Args:
            use_dense: True = Dense scheduler (for FC layers like FC1)
                       False = BSR scheduler (for sparse conv layers)
        """
        bsr_config = self._csr_read(CSRMap.BSR_CONFIG)
        if use_dense:
            bsr_config |= CSRMap.SCHED_MODE_DENSE  # Set bit 0
        else:
            bsr_config &= ~CSRMap.SCHED_MODE_DENSE  # Clear bit 0
        self._csr_write(CSRMap.BSR_CONFIG, bsr_config)

    # =========================================================================
    # Sparsity-aware scheduler selection
    # =========================================================================

    # --- Tuneable thresholds (edit these after profiling on real hardware) ---
    SPARSITY_THRESHOLD = 50.0   # Block-sparsity % above which BSR wins
    MIN_BLOCKS_FOR_BSR = 4      # Minimum total blocks for BSR to be worthwhile

    @staticmethod
    def should_use_sparse(weight_2d: np.ndarray,
                          block_size: int = 14,
                          sparsity_threshold: float = None,
                          min_blocks: int = None) -> Tuple[bool, float]:
        """
        Decide whether a layer should run through the BSR (sparse) scheduler
        or the dense GEMM scheduler on the 14×14 systolic array.

        Decision criteria
        -----------------
        1. **Matrix too small** — if the weight matrix tiles into fewer than
           ``MIN_BLOCKS_FOR_BSR`` total blocks, the fixed overhead of loading
           BSR metadata (row_ptr, col_idx) dominates.  → use dense.
        2. **Sparsity below threshold** — each BSR block incurs per-block
           bookkeeping (~15 extra cycles for metadata fetch + indexing).
           Below ~50 % block-sparsity the savings from skipping zero blocks
           are smaller than that overhead.  → use dense.
        3. Otherwise → use BSR sparse.

        The 50 % default is conservative; on real hardware you can profile and
        lower it (40 % is common on larger systolic arrays).

        Parameters
        ----------
        weight_2d : np.ndarray
            2-D weight matrix (already reshaped from Conv4D if needed).
        block_size : int
            Hardware block dimension (default 14).
        sparsity_threshold : float, optional
            Override ``SPARSITY_THRESHOLD`` for this call.
        min_blocks : int, optional
            Override ``MIN_BLOCKS_FOR_BSR`` for this call.

        Returns
        -------
        use_sparse : bool
            True  → select BSR scheduler (``set_scheduler_mode(use_dense=False)``)
            False → select dense scheduler (``set_scheduler_mode(use_dense=True)``)
        block_sparsity : float
            Measured block-sparsity percentage (0–100).
        """
        if sparsity_threshold is None:
            sparsity_threshold = AccelDriver.SPARSITY_THRESHOLD
        if min_blocks is None:
            min_blocks = AccelDriver.MIN_BLOCKS_FOR_BSR

        rows, cols = weight_2d.shape
        block_rows = -(-rows // block_size)   # ceil division
        block_cols = -(-cols // block_size)
        total_blocks = block_rows * block_cols

        # --- Criterion 1: matrix too small ---
        if total_blocks < min_blocks:
            return False, 0.0

        # --- Count zero blocks ---
        zero_blocks = 0
        for br in range(block_rows):
            for bc in range(block_cols):
                r0 = br * block_size
                c0 = bc * block_size
                block = weight_2d[r0:min(r0 + block_size, rows),
                                  c0:min(c0 + block_size, cols)]
                if np.all(block == 0):
                    zero_blocks += 1

        block_sparsity = 100.0 * zero_blocks / total_blocks

        # --- Criterion 2: sparsity too low ---
        if block_sparsity < sparsity_threshold:
            return False, block_sparsity

        return True, block_sparsity

    def auto_select_scheduler(self, weight_2d: np.ndarray,
                              layer_name: str = "") -> bool:
        """
        Convenience wrapper: measure block-sparsity of *weight_2d*, pick the
        best scheduler, program the CSR, and return the choice.

        Parameters
        ----------
        weight_2d : np.ndarray
            2-D weight matrix for this layer.
        layer_name : str
            Optional label for logging.

        Returns
        -------
        use_sparse : bool
            True if BSR scheduler was selected.
        """
        use_sparse, sparsity = self.should_use_sparse(weight_2d, self.BLOCK_SIZE)
        self.set_scheduler_mode(use_dense=not use_sparse)

        mode_str = "BSR-sparse" if use_sparse else "Dense-GEMM"
        tag = f" [{layer_name}]" if layer_name else ""
        print(f"Scheduler{tag}: {mode_str}  (block-sparsity {sparsity:.1f}%)")
        return use_sparse
    
    # =========================================================================
    # Private methods
    # =========================================================================
    
    def _csr_write(self, addr: int, value: int):
        """Write to CSR register."""
        if self.simulation:
            self.csr.write(addr, value)
        else:
            self.csr.write(self.csr_base + addr, value)
            
    def _csr_read(self, addr: int) -> int:
        """Read from CSR register."""
        if self.simulation:
            return self.csr.read(addr)
        else:
            return self.csr.read(self.csr_base + addr)
    
    def _wait_dma_done(self, ctrl_addr: int, timeout_ms: int = 500):
        """Wait for DMA completion."""
        start = time.time()
        timeout_s = timeout_ms / 1000.0
        
        while True:
            ctrl = self._csr_read(ctrl_addr)
            done = (ctrl >> 2) & 0x1
            
            if done:
                return
                
            if time.time() - start > timeout_s:
                raise TimeoutError(f"DMA timeout on {ctrl_addr:#x}")
                
            time.sleep(0.0001)  # 100us poll


class SimulatedCSR:
    """Simulated CSR interface for testing without hardware."""
    
    def __init__(self):
        self.regs = {}
        # Initialize status as done after any operation
        self.regs[CSRMap.STATUS] = 0x2  # done=1
        self.regs[CSRMap.DMA_CTRL] = 0x4  # done=1
        self.regs[CSRMap.ACT_DMA_CTRL] = 0x4  # done=1
        
    def write(self, addr: int, value: int):
        self.regs[addr] = value
        
        # Simulate start → done transition
        if addr == CSRMap.CTRL and (value & 0x1):
            self.regs[CSRMap.STATUS] = 0x2  # done
            self.regs[CSRMap.PERF_TOTAL] = 1000
            self.regs[CSRMap.PERF_ACTIVE] = 800
            
        if addr in [CSRMap.DMA_CTRL, CSRMap.ACT_DMA_CTRL] and (value & 0x1):
            self.regs[addr] = 0x4  # done
            
    def read(self, addr: int) -> int:
        return self.regs.get(addr, 0)


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    print("ACCEL-v1 Driver Test (Simulation Mode)")
    print("=" * 50)
    
    # Create driver in simulation mode
    accel = AccelDriver(simulation=True)
    
    # Configure for 16x32 x 32x16 = 16x16 output
    M, N, K = 16, 16, 32
    accel.configure_dimensions(M, N, K)
    print(f"Configured: M={M}, N={N}, K={K}")
    
    # Create sparse weights (2 blocks)
    num_blocks = 2
    row_ptr = np.array([0, 2], dtype=np.int32)  # Row 0 has 2 blocks
    col_idx = np.array([0, 1], dtype=np.int16)  # Columns 0 and 1
    weights = np.random.randint(-128, 127, (num_blocks, 14, 14), dtype=np.int8)
    
    bytes_loaded = accel.load_sparse_weights(row_ptr, col_idx, weights)
    print(f"Loaded {bytes_loaded} bytes of sparse weights")
    
    # Create dense activations
    activations = np.random.randint(-128, 127, (M, K), dtype=np.int8)
    bytes_loaded = accel.load_activations(activations)
    print(f"Loaded {bytes_loaded} bytes of activations")
    
    # Run inference
    success, results = accel.run_inference()
    print(f"Inference {'succeeded' if success else 'failed'}")
    print(f"  Total cycles: {results['cycles']}")
    print(f"  Active cycles: {results['active_cycles']}")
    print(f"  Utilization: {results['utilization']:.1f}%")
    print(f"  Sample results: {results['result_sample']}")
