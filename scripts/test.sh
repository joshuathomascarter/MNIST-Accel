#!/bin/bash
# =============================================================================
# test.sh - Unified Test Runner for ACCEL Project
# =============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
SIM_DIR="$BUILD_DIR/sim"
LOG_DIR="$BUILD_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
}

# =============================================================================
# Test: Python AXI Simulator
# =============================================================================
test_python() {
    print_header "Test 1: Python AXI Simulator"
    
    cd "$PROJECT_ROOT/accel/python/host"
    
    print_info "Running Python tests..."
    python3 axi_master_sim.py
    
    if [ $? -eq 0 ]; then
        print_success "Python AXI simulator tests passed"
    else
        print_error "Python tests failed"
        exit 1
    fi
}

# =============================================================================
# Test: Verilog AXI Testbench
# =============================================================================
test_verilog() {
    print_header "Test 2: Verilog AXI Testbench"
    
    cd "$PROJECT_ROOT"
    
    # Build if needed
    if [ ! -f "$SIM_DIR/tb_axi.vvp" ]; then
        print_info "Building AXI testbench first..."
        "$PROJECT_ROOT/scripts/build.sh" axi
    fi
    
    print_info "Running Verilog simulation..."
    vvp "$SIM_DIR/tb_axi.vvp" 2>&1 | tee "$LOG_DIR/test_axi.log"
    
    if grep -q "ALL PASSED" "$LOG_DIR/test_axi.log"; then
        print_success "Verilog AXI testbench passed"
    else
        print_error "Verilog testbench failed"
        tail -30 "$LOG_DIR/test_axi.log"
        exit 1
    fi
}

# =============================================================================
# Test: Cocotb Integration
# =============================================================================
test_cocotb() {
    print_header "Test 3: Cocotb Integration Tests"
    
    cd "$PROJECT_ROOT/testbench/cocotb"
    
    if [ ! -f "Makefile.cocotb" ]; then
        print_info "Cocotb tests not available, skipping..."
        return 0
    fi
    
    print_info "Running Cocotb tests..."
    make -f Makefile.cocotb 2>&1 | tee "$LOG_DIR/test_cocotb.log"
    
    if [ $? -eq 0 ]; then
        print_success "Cocotb tests passed"
    else
        print_error "Cocotb tests failed"
        exit 1
    fi
}

# =============================================================================
# Main
# =============================================================================
mkdir -p "$LOG_DIR"

case "${1:-all}" in
    python)
        test_python
        ;;
    verilog)
        test_verilog
        ;;
    cocotb)
        test_cocotb
        ;;
    all)
        test_python
        test_verilog
        # test_cocotb  # Enable when ready
        ;;
    *)
        echo "Usage: $0 {python|verilog|cocotb|all}"
        echo "  python   - Run Python unit tests"
        echo "  verilog  - Run Verilog testbench"
        echo "  cocotb   - Run Cocotb integration tests"
        echo "  all      - Run all tests (default)"
        exit 1
        ;;
esac

print_header "All Tests Complete!"
print_success "✓ Test suite passed"
