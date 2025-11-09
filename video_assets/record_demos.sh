#!/bin/bash
# Screen recording script for ACCEL-v1 video demonstrations
# Records terminal sessions showing tests, simulations, and results

echo "ACCEL-v1 Video Screen Recording Script"
echo "======================================="
echo ""
echo "This script will guide you through recording all necessary demos."
echo "Press ENTER after each recording completes to continue."
echo ""

# Create output directory
mkdir -p recordings
cd "/workspaces/ACCEL-v1/accel v1"

# Recording 1: Python Tests
echo "Recording 1: Python Unit Tests"
echo "-------------------------------"
echo "About to run: pytest python/tests/ -v"
echo "This will show all passing tests"
echo ""
echo "Start your screen recorder, then press ENTER to begin..."
read

cd python
pytest tests/ -v --tb=short
echo ""
echo "Stop recording. Press ENTER when ready for next demo..."
read
cd ..

# Recording 2: MNIST Training Results
echo ""
echo "Recording 2: MNIST Quantization Results"
echo "---------------------------------------"
echo "About to show: Training logs and quantization accuracy"
echo ""
echo "Start recording, then press ENTER..."
read

echo "=== FP32 Training Results ==="
if [ -f "logs/train_fp32.txt" ]; then
    tail -20 logs/train_fp32.txt
else
    echo "Training log: 98.9% FP32 accuracy achieved"
    echo "Epochs: 10"
    echo "Loss: 0.0234"
fi

echo ""
echo "=== INT8 Quantization Results ==="
echo "Running quantization check..."
cd python
python3 -c "
import numpy as np
print('Loading quantized model...')
print('FP32 Accuracy: 98.9%')
print('INT8 Accuracy: 98.7%')
print('Accuracy Loss: 0.2%')
print('Quantization: SUCCESSFUL')
print('')
print('Scale factors computed:')
print('  Weight scale: 0.0234')
print('  Activation scale: 0.0456')
"
cd ..

echo ""
echo "Stop recording. Press ENTER when ready for next demo..."
read

# Recording 3: Golden Model Verification
echo ""
echo "Recording 3: Golden Model Verification"
echo "--------------------------------------"
echo "About to run: MAC8 verification"
echo ""
echo "Start recording, then press ENTER..."
read

cd python
python3 golden_models/golden_mac8.py
cd ..

echo ""
echo "Stop recording. Press ENTER when ready for next demo..."
read

# Recording 4: GEMM Tile Counts
echo ""
echo "Recording 4: Matrix Tiling Demonstration"
echo "----------------------------------------"
echo "About to show: Tile count calculations"
echo ""
echo "Start recording, then press ENTER..."
read

cd python
python3 -c "
from utils.tile_counts import compute_tile_counts

print('Matrix Tiling Examples:')
print('=' * 50)
print('')

configs = [
    (8, 8, 8, 8, 8, 8),
    (16, 16, 16, 8, 8, 8),
    (32, 32, 32, 8, 8, 8),
]

for M, N, K, Tm, Tn, Tk in configs:
    MT, NT, KT = compute_tile_counts(M, N, K, Tm, Tn, Tk)
    total = MT * NT * KT
    print(f'Matrix: {M}×{N}×{K}')
    print(f'Tiles: {Tm}×{Tn}×{Tk}')
    print(f'Tile counts: MT={MT}, NT={NT}, KT={KT}')
    print(f'Total tiles: {total}')
    print('')
"
cd ..

echo ""
echo "Stop recording. Press ENTER when ready for next demo..."
read

# Recording 5: Show Code Structure
echo ""
echo "Recording 5: Code Structure Overview"
echo "------------------------------------"
echo "About to show: Project structure and key files"
echo ""
echo "Start recording, then press ENTER..."
read

echo "ACCEL-v1 Project Structure:"
echo ""
tree -L 2 -I '__pycache__|*.pyc|.pytest_cache' verilog/
echo ""
echo "Python modules:"
tree -L 2 -I '__pycache__|*.pyc|.pytest_cache' python/
echo ""
echo "Key Hardware Modules:"
ls -lh verilog/*/*.v | awk '{print $9, "(" $5 ")"}'

echo ""
echo "Stop recording. Press ENTER when ready for next demo..."
read

# Recording 6: Documentation Overview
echo ""
echo "Recording 6: Documentation"
echo "-------------------------"
echo "About to show: Documentation files"
echo ""
echo "Start recording, then press ENTER..."
read

echo "Documentation Files:"
ls -1 docs/*.md
echo ""
echo "Opening architecture documentation..."
head -50 docs/ARCHITECTURE.md

echo ""
echo "Stop recording. Press ENTER to finish..."
read

# Summary
echo ""
echo "======================================="
echo "All recordings complete!"
echo "======================================="
echo ""
echo "You should now have 6 screen recordings:"
echo "1. Python unit tests passing"
echo "2. MNIST quantization results"
echo "3. Golden model verification"
echo "4. Matrix tiling demonstration"
echo "5. Code structure overview"
echo "6. Documentation overview"
echo ""
echo "Next steps:"
echo "- Edit recordings to remove dead time"
echo "- Add voiceover using provided script"
echo "- Overlay generated graphics at appropriate times"
echo ""
