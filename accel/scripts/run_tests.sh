#!/bin/bash
make -C ../tests
if [ $? -eq 0 ]; then
    ./tests/unit/test_mac8
    ./tests/unit/test_pe
    ./tests/unit/test_csr
    ./tests/integration/test_systolic_array
else
    echo "Compilation failed."
fi