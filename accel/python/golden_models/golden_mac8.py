"""golden_mac8.py - Golden checker for mac8 testbench CSV
Reads tb/unit/mac8_results.csv produced by tb_mac8.v and reproduces the INT8xINT8
multiply-accumulate into a 32-bit signed accumulator with optional bypass (residual).

NEW: Supports bypass mode (0=MAC, 1=Adder for ResNet skip connections)
"""

import csv
import sys
import argparse

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


# ============================================================================
# UTILITY FUNCTIONS (Bit Operations)
# ============================================================================

def to_s32(x):
    """
    Convert a Python integer to a signed 32-bit value.
    
    Why? Python integers are unlimited precision. But hardware registers
    are finite (32-bit). This function simulates that wraparound.
    
    Hardware analogy:
      - If you overflow a 32-bit register, it wraps around.
      - Example: 0xFFFFFFFF + 1 = 0x100000000 (loses the top bit) = 0x00000000
    """
    x &= 0xFFFFFFFF                    # Mask to 32 bits
    return x - 0x100000000 if x & 0x80000000 else x  # Sign-extend


def interpret_8bit_as_signed(val):
    """Interpret an 8-bit value as signed instead of unsigned"""
    val &= 0xFF
    return val - 0x100 if val & 0x80 else val


# ============================================================================
# GOLDEN LOGIC (The Reference Circuit)
# ============================================================================

class GoldenMAC8:
    """
    A reference implementation of mac8.sv in Python.
    
    This is like a SOFTWARE SIMULATION of your hardware.
    It takes the same inputs and produces the same outputs.
    """
    
    def __init__(self):
        self.acc = 0  # Internal register (initialized to 0)
    
    def cycle(self, a, b, bypass, en, clr):
        """
        Simulate one clock cycle.
        
        Inputs:
          - a:      Activation (8-bit)
          - b:      Weight (8-bit)
          - bypass: Control (0=MAC, 1=Adder)
          - en:     Enable (0=hold, 1=update)
          - clr:    Clear (1=reset)
        
        Output:
          - acc:    Updated accumulator value
        
        ALGORITHM (matches mac8.sv exactly):
        """
        
        # ─────────────────────────────────────────────────────────
        # STEP 1: CLEAR (highest priority)
        # ─────────────────────────────────────────────────────────
        if clr:
            self.acc = 0
            return self.acc
        
        # ─────────────────────────────────────────────────────────
        # STEP 2: COMPUTE (if enabled)
        # ─────────────────────────────────────────────────────────
        if not en:
            # If not enabled, accumulator holds (no change)
            return self.acc
        
        # At this point: en=1, clr=0
        # We will update the accumulator. But HOW depends on bypass.
        
        if bypass == 0:
            # ─────────────────────────────────────────────────────
            # MAC MODE (bypass=0): Multiply-Accumulate
            # ─────────────────────────────────────────────────────
            # acc_new = acc_old + (a × b)
            
            prod = a * b                    # Multiply (signed 8×8 = 16-bit)
            sum_raw = self.acc + prod       # Add to accumulator
            self.acc = to_s32(sum_raw)      # Wrap to 32-bit signed
            
        else:  # bypass == 1
            # ─────────────────────────────────────────────────────
            # BYPASS MODE (bypass=1): Residual Addition
            # ─────────────────────────────────────────────────────
            # acc_new = acc_old + a  (skip the multiply)
            # Used for skip connections in ResNet
            
            a_extended = sign_extend_8_to_32(a)  # Convert 8-bit to 32-bit
            sum_raw = self.acc + a_extended      # Add to accumulator
            self.acc = to_s32(sum_raw)           # Wrap to 32-bit signed
        
        return self.acc


# ============================================================================
# VERIFICATION: Compare Golden vs. RTL
# ============================================================================

def verify_mac8(csv_path, verbose=False):
    """
    Read a CSV file (produced by Verilator) and compare each cycle.
    
    FLOW:
      1. For each row in CSV (each cycle):
         - Extract input signals (a, b, bypass, en, clr)
         - Extract RTL output (acc_rtl)
      
      2. Run golden logic:
         - Feed same inputs to GoldenMAC8
         - Get golden output (acc_golden)
      
      3. Compare:
         - If acc_rtl == acc_golden → ✓ PASS
         - If acc_rtl != acc_golden → ✗ FAIL (bug in RTL)
    """
    
    golden = GoldenMAC8()
    errors = []
    
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            cycle = int(row["cycle"])
            a = int(row["a"])
            b = int(row["b"])
            bypass = int(row.get("bypass", 0))
            en = int(row["en"])
            clr = int(row["clr"])
            acc_rtl = int(row["acc"])
            
            # Run golden model for this cycle
            acc_golden = golden.cycle(a, b, bypass, en, clr)
            
            # Check if it matches
            if acc_rtl != acc_golden:
                error_msg = (
                    f"MISMATCH at cycle {cycle}: "
                    f"a={a} b={b} bypass={bypass} en={en} clr={clr} | "
                    f"RTL={acc_rtl} Golden={acc_golden}"
                )
                errors.append(error_msg)
                
                if verbose:
                    print(f"  ✗ {error_msg}")
    
    # ─────────────────────────────────────────────────────────
    # REPORT RESULTS
    # ─────────────────────────────────────────────────────────
    if not errors:
        print("✓ PASS: All cycles match golden model")
        return 0
    else:
        print(f"✗ FAIL: {len(errors)} cycle(s) mismatch")
        for err in errors:
            print(f"  {err}")
        return len(errors)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify MAC8 RTL against golden model"
    )
    parser.add_argument(
        "--csv",
        default="tb/unit/mac8_results.csv",
        help="Path to CSV file from testbench"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error messages"
    )
    
    args = parser.parse_args()
    
    try:
        errors = verify_mac8(args.csv, verbose=args.verbose)
        sys.exit(0 if errors == 0 else 1)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)


if __name__ == "__main__":
    main()
