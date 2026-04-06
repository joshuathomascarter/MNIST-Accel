#!/bin/zsh
cd /Users/joshcarter/MNIST-Accel/hw/sim/sv

echo "=== Step 1: Verilator compilation ===" 2>&1
verilator --sv --binary --timing --Wno-fatal --Wno-PINMISSING --Wno-UNDRIVEN --Wno-UNOPTFLAT -f filelist.f --top-module tb_soc_top -o obj_dir/tb_soc_top 2>&1
COMPILE_EXIT=$?
echo "COMPILE_EXIT_CODE=$COMPILE_EXIT"

if [ $COMPILE_EXIT -eq 0 ]; then
    echo ""
    echo "=== Step 2: Running simulation ==="
    ./obj_dir/tb_soc_top 2>&1
    SIM_EXIT=$?
    echo "SIM_EXIT_CODE=$SIM_EXIT"
else
    echo "Compilation failed — skipping simulation run."
fi
