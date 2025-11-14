#!/bin/bash
# =============================================================================
# ci_verilator.sh — Continuous Integration with Verilator
# =============================================================================
# Purpose:
#   Automated regression testing for ACCEL-v1 RTL modules.
#   Compiles and runs testbenches for DMA, metadata, and integration paths.
#
# Usage:
#   ./ci_verilator.sh [test_name]
#   ./ci_verilator.sh all
#
# =============================================================================

set -e

WORKSPACE="/workspaces/ACCEL-v1/accel v1"
BUILD_DIR="${WORKSPACE}/../build"
LOG_DIR="${WORKSPACE}/../logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}ACCEL-v1 CI: Verilator Regression Suite${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# ============================================================================
# Helper Functions
# ============================================================================

run_test() {
    local test_name=$1
    local test_dir=$2
    local rtl_files=$3
    
    echo -ne "${YELLOW}[TEST]${NC} $test_name ... "
    
    mkdir -p "$BUILD_DIR/$test_name"
    cd "$BUILD_DIR/$test_name"
    
    # Compile with Verilator
    if verilator -Wall -Wno-MODDUP -Wno-WIDTHEXPAND \
        -I"${WORKSPACE}/verilog" \
        -I"${WORKSPACE}/verilog/dma" \
        -I"${WORKSPACE}/verilog/meta" \
        -I"${WORKSPACE}/verilog/top" \
        --trace \
        --cc \
        $rtl_files \
        2>&1 | tee "$LOG_DIR/$test_name.compile.log"; then
        
        # Make C++ testbench
        cd obj_dir
        make -f Vtest.mk 2>&1 | tee "$LOG_DIR/$test_name.build.log"
        
        # Run simulation
        if ./Vtest 2>&1 | tee "$LOG_DIR/$test_name.sim.log"; then
            echo -e "${GREEN}PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${RED}FAIL${NC} (simulation error)"
            ((TESTS_FAILED++))
            FAILED_TESTS+=("$test_name")
            return 1
        fi
    else
        echo -e "${RED}FAIL${NC} (compile error)"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$test_name")
        return 1
    fi
}

# ============================================================================
# Setup
# ============================================================================

mkdir -p "$BUILD_DIR" "$LOG_DIR"

# Check for Verilator
if ! command -v verilator &> /dev/null; then
    echo -e "${RED}ERROR: Verilator not installed${NC}"
    exit 1
fi

VERILATOR_VERSION=$(verilator --version 2>&1 | head -1)
echo -e "Verilator: ${GREEN}${VERILATOR_VERSION}${NC}\n"

# ============================================================================
# Test Suite
# ============================================================================

TEST_SUITE=$1

# Test 1: DMA Lite Module
if [[ "$TEST_SUITE" == "all" || "$TEST_SUITE" == "dma_lite" ]]; then
    run_test "test_dma_lite" \
        "${WORKSPACE}/tb/unit" \
        "${WORKSPACE}/verilog/dma/dma_lite.v ${WORKSPACE}/tb/unit/tb_dma_lite.sv" \
        || true
fi

# Test 2: Metadata Decoder
if [[ "$TEST_SUITE" == "all" || "$TEST_SUITE" == "meta_decode" ]]; then
    run_test "test_meta_decode" \
        "${WORKSPACE}/tb/unit" \
        "${WORKSPACE}/verilog/meta/meta_decode.sv ${WORKSPACE}/tb/unit/tb_meta_decode.sv" \
        || true
fi

# Test 3: BSR DMA (existing)
if [[ "$TEST_SUITE" == "all" || "$TEST_SUITE" == "bsr_dma" ]]; then
    run_test "test_bsr_dma" \
        "${WORKSPACE}/tb/unit" \
        "${WORKSPACE}/verilog/dma/bsr_dma.v ${WORKSPACE}/tb/unit/tb_bsr_dma.sv" \
        || true
fi

# Test 4: AXI Host Interface
if [[ "$TEST_SUITE" == "all" || "$TEST_SUITE" == "axi_host" ]]; then
    # Note: Requires full accel_top dependencies; skip in basic CI
    echo -e "${YELLOW}[SKIP]${NC} test_axi_host (requires full module stack)"
fi

# Test 5: Integration (DMA → Metadata → Scheduler skeleton)
if [[ "$TEST_SUITE" == "all" || "$TEST_SUITE" == "integration" ]]; then
    # TODO: Build integration testbench when scheduler available
    echo -e "${YELLOW}[SKIP]${NC} test_integration (scheduler not finalized)"
fi

# ============================================================================
# Summary
# ============================================================================

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}Test Summary${NC}"
echo -e "${YELLOW}========================================${NC}\n"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo -e "Total:  ${TOTAL_TESTS}"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  • $test"
        echo -e "    Logs: $LOG_DIR/$test.*.log"
    done
    exit 1
else
    echo -e "\n${GREEN}✓ All tests passed!${NC}\n"
    exit 0
fi
