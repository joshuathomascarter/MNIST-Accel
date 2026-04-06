#!/usr/bin/env python3
"""
power_profiling.py — Power Analysis for ACCEL-v1 on Zynq
=========================================================

Provides power measurement and profiling utilities for:
  - Real-time power monitoring via Zynq XADC
  - Energy-per-inference calculations
  - Power efficiency metrics (GOPS/W)
  - Thermal monitoring

Requires: PYNQ framework on Zynq board

Author: Joshua Carter
Date: December 2024
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading

# Try PYNQ imports
try:
    from pynq import Overlay
    from pynq.lib import AxiGPIO
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False


@dataclass
class PowerSample:
    """Single power measurement sample."""
    timestamp: float      # Time in seconds
    vccint: float         # Core voltage (V)
    vccaux: float         # Auxiliary voltage (V)
    vccbram: float        # BRAM voltage (V)
    current_ma: float     # Estimated current (mA)
    power_mw: float       # Estimated power (mW)
    temperature: float    # Die temperature (°C)


@dataclass  
class PowerProfile:
    """Power profile for an inference run."""
    name: str
    duration_ms: float
    samples: List[PowerSample]
    
    @property
    def avg_power_mw(self) -> float:
        if not self.samples:
            return 0.0
        return np.mean([s.power_mw for s in self.samples])
    
    @property
    def peak_power_mw(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.power_mw for s in self.samples)
    
    @property
    def energy_mj(self) -> float:
        """Total energy in millijoules."""
        return self.avg_power_mw * self.duration_ms / 1000.0
    
    @property
    def avg_temperature(self) -> float:
        if not self.samples:
            return 0.0
        return np.mean([s.temperature for s in self.samples])


class XADCMonitor:
    """
    XADC-based power monitor for Zynq-7000.
    
    Uses the Zynq's built-in ADC to measure:
    - VCCINT: Core voltage
    - VCCAUX: Auxiliary voltage  
    - VCCBRAM: BRAM voltage
    - Temperature
    
    Power is estimated based on voltage and typical current profiles.
    """
    
    # XADC register addresses (Zynq-7000)
    XADC_BASE = 0xF8007100
    TEMP_REG = 0x00
    VCCINT_REG = 0x04
    VCCAUX_REG = 0x08
    VCCBRAM_REG = 0x18
    
    # Conversion constants
    TEMP_SCALE = 503.975 / 65536  # °C per LSB
    TEMP_OFFSET = -273.15
    VOLT_SCALE = 3.0 / 65536      # V per LSB
    
    # Typical current profiles for Zynq-7020 (mA)
    CURRENT_IDLE = 200
    CURRENT_ACTIVE = 800
    CURRENT_PL_BASE = 100  # PL baseline
    CURRENT_PL_PER_LUT = 0.01  # mA per LUT used
    
    def __init__(self, simulation: bool = False):
        """
        Initialize XADC monitor.
        
        Args:
            simulation: Use simulated values
        """
        self.simulation = simulation or not PYNQ_AVAILABLE
        self._mmio = None
        
        if not self.simulation:
            try:
                from pynq import MMIO
                self._mmio = MMIO(self.XADC_BASE, 0x100)
            except Exception as e:
                print(f"[PowerProfiler] Warning: XADC init failed: {e}")
                self.simulation = True
    
    def read_sample(self) -> PowerSample:
        """Read current power/temperature sample."""
        if self.simulation:
            return self._simulated_sample()
        
        try:
            # Read raw ADC values
            temp_raw = self._mmio.read(self.TEMP_REG) >> 4
            vccint_raw = self._mmio.read(self.VCCINT_REG) >> 4
            vccaux_raw = self._mmio.read(self.VCCAUX_REG) >> 4
            vccbram_raw = self._mmio.read(self.VCCBRAM_REG) >> 4
            
            # Convert to physical units
            temp = temp_raw * self.TEMP_SCALE + self.TEMP_OFFSET
            vccint = vccint_raw * self.VOLT_SCALE
            vccaux = vccaux_raw * self.VOLT_SCALE
            vccbram = vccbram_raw * self.VOLT_SCALE
            
            # Estimate current and power
            current = self.CURRENT_ACTIVE + self.CURRENT_PL_BASE
            power = vccint * current + vccaux * 50 + vccbram * 20
            
            return PowerSample(
                timestamp=time.time(),
                vccint=vccint,
                vccaux=vccaux,
                vccbram=vccbram,
                current_ma=current,
                power_mw=power,
                temperature=temp
            )
        except Exception as e:
            print(f"[PowerProfiler] Read error: {e}")
            return self._simulated_sample()
    
    def _simulated_sample(self) -> PowerSample:
        """Generate simulated power sample."""
        # Simulate realistic Zynq values
        return PowerSample(
            timestamp=time.time(),
            vccint=1.0 + np.random.normal(0, 0.01),
            vccaux=1.8 + np.random.normal(0, 0.02),
            vccbram=1.0 + np.random.normal(0, 0.01),
            current_ma=500 + np.random.normal(0, 50),
            power_mw=800 + np.random.normal(0, 100),
            temperature=45 + np.random.normal(0, 2)
        )


class PowerProfiler:
    """
    Power profiler for ACCEL-v1 inference runs.
    
    Usage:
        profiler = PowerProfiler()
        
        with profiler.profile("inference"):
            accel.run_inference()
            
        print(profiler.last_profile)
    """
    
    def __init__(self, sample_rate_hz: float = 100.0, simulation: bool = False):
        """
        Initialize power profiler.
        
        Args:
            sample_rate_hz: Sampling rate for power measurements
            simulation: Use simulated measurements
        """
        self.sample_rate = sample_rate_hz
        self.sample_interval = 1.0 / sample_rate_hz
        self.simulation = simulation
        
        self.xadc = XADCMonitor(simulation=simulation)
        self.profiles: List[PowerProfile] = []
        self.last_profile: Optional[PowerProfile] = None
        
        self._sampling = False
        self._samples: List[PowerSample] = []
        self._sample_thread: Optional[threading.Thread] = None
    
    def start_sampling(self):
        """Start background power sampling."""
        self._sampling = True
        self._samples = []
        self._sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sample_thread.start()
    
    def stop_sampling(self) -> List[PowerSample]:
        """Stop sampling and return collected samples."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
        return self._samples.copy()
    
    def _sample_loop(self):
        """Background sampling loop."""
        while self._sampling:
            sample = self.xadc.read_sample()
            self._samples.append(sample)
            time.sleep(self.sample_interval)
    
    def profile(self, name: str):
        """
        Context manager for profiling a code block.
        
        Usage:
            with profiler.profile("inference"):
                run_inference()
        """
        return ProfileContext(self, name)
    
    def create_profile(self, name: str, start_time: float, 
                      samples: List[PowerSample]) -> PowerProfile:
        """Create a power profile from samples."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        profile = PowerProfile(
            name=name,
            duration_ms=duration_ms,
            samples=samples
        )
        
        self.profiles.append(profile)
        self.last_profile = profile
        
        return profile
    
    def get_efficiency_metrics(self, profile: PowerProfile, 
                               operations: int) -> Dict[str, float]:
        """
        Calculate efficiency metrics for a profile.
        
        Args:
            profile: PowerProfile to analyze
            operations: Number of operations (MACs * 2)
            
        Returns:
            Dictionary with efficiency metrics
        """
        gops = operations / (profile.duration_ms * 1e6)
        gops_per_watt = gops / (profile.avg_power_mw / 1000) if profile.avg_power_mw > 0 else 0
        ops_per_joule = operations / (profile.energy_mj / 1000) if profile.energy_mj > 0 else 0
        
        return {
            'throughput_gops': gops,
            'power_mw': profile.avg_power_mw,
            'energy_mj': profile.energy_mj,
            'efficiency_gops_per_w': gops_per_watt,
            'ops_per_joule': ops_per_joule,
            'temperature_c': profile.avg_temperature,
        }
    
    def print_profile_summary(self, profile: PowerProfile, operations: int = 0):
        """Print formatted profile summary."""
        print(f"\n{'='*60}")
        print(f"Power Profile: {profile.name}")
        print(f"{'='*60}")
        print(f"  Duration:      {profile.duration_ms:.2f} ms")
        print(f"  Samples:       {len(profile.samples)}")
        print(f"  Avg Power:     {profile.avg_power_mw:.1f} mW")
        print(f"  Peak Power:    {profile.peak_power_mw:.1f} mW")
        print(f"  Energy:        {profile.energy_mj:.3f} mJ")
        print(f"  Temperature:   {profile.avg_temperature:.1f} °C")
        
        if operations > 0:
            metrics = self.get_efficiency_metrics(profile, operations)
            print(f"\n  Throughput:    {metrics['throughput_gops']:.2f} GOPS")
            print(f"  Efficiency:    {metrics['efficiency_gops_per_w']:.2f} GOPS/W")
            print(f"  Ops/Joule:     {metrics['ops_per_joule']:.2e}")


class ProfileContext:
    """Context manager for power profiling."""
    
    def __init__(self, profiler: PowerProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = 0
        
    def __enter__(self):
        self.start_time = time.time()
        self.profiler.start_sampling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        samples = self.profiler.stop_sampling()
        self.profiler.create_profile(self.name, self.start_time, samples)
        return False


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    print("Power Profiling Test (Simulation Mode)")
    print("=" * 60)
    
    profiler = PowerProfiler(sample_rate_hz=100, simulation=True)
    
    # Simulate an inference run
    with profiler.profile("test_inference"):
        time.sleep(0.1)  # Simulate 100ms inference
    
    # Print results
    profile = profiler.last_profile
    operations = 2 * 16 * 16 * 32  # Example: 16x32 @ 32x16
    
    profiler.print_profile_summary(profile, operations)
    
    print("\nRaw samples (first 5):")
    for s in profile.samples[:5]:
        print(f"  t={s.timestamp:.3f}: {s.power_mw:.1f}mW, {s.temperature:.1f}°C")
