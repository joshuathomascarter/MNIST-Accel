#!/bin/bash
# =============================================================================
# build.sh - Unified Build Script for ACCEL Project
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
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create build directories
mkdir -p "$SIM_DIR" "$LOG_DIR"

# =============================================================================
# Build AXI Testbench
# =============================================================================
build_axi() {
    print_header "Building AXI Testbench"
    
    cd "$PROJECT_ROOT"
    
    print_info "Compiling with iverilog..."
    iverilog -g2009 -Wall -Winfloop \
        -o "$SIM_DIR/tb_axi.vvp" \
        rtl/host_iface/axi_lite_slave_v2.sv \
        rtl/host_iface/tb_axi_lite_slave_enhanced.sv \
        2>&1 | tee "$LOG_DIR/build_axi.log"
    
    print_success "AXI testbench built: $SIM_DIR/tb_axi.vvp"
}

# =============================================================================
# Build Top-Level Design
# =============================================================================
build_top() {
    print_header "Building Top-Level Design"
    
    cd "$PROJECT_ROOT"
    
    print_info "Compiling accel_top..."
    iverilog -g2009 -Wall -Winfloop \
        -I rtl \
        -o "$SIM_DIR/accel_top.vvp" \
        rtl/top/accel_top.v \
        2>&1 | tee "$LOG_DIR/build_top.log"
    
    print_success "Top-level design built: $SIM_DIR/accel_top.vvp"
}

# =============================================================================
# Main
# =============================================================================
case "${1:-all}" in
    axi)
        build_axi
        ;;
    top)
        build_top
        ;;
    all)
        build_axi
        build_top
        ;;
    *)
        echo "Usage: $0 {axi|top|all}"
        echo "  axi  - Build AXI testbench only"
        echo "  top  - Build top-level design only"
        echo "  all  - Build everything (default)"
        exit 1
        ;;
esac

print_success "Build complete!"
