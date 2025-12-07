#!/bin/bash
# =============================================================================
# convert_to_sv.sh - Convert all .v files to .sv (SystemVerilog standard)
# =============================================================================
# Rationale: Industry standard (AMD/Tenstorrent) uses .sv exclusively for:
#   - SVA assertions (assert property)
#   - Interfaces and modports
#   - Structs and enums
#   - Advanced verification constructs
# =============================================================================

set -e

echo "=========================================="
echo "Converting .v files to .sv (SystemVerilog)"
echo "=========================================="

# Find all .v files in rtl/
v_files=$(find rtl -name "*.v" -type f)

if [ -z "$v_files" ]; then
    echo "No .v files found - already converted!"
    exit 0
fi

echo "Found $(echo "$v_files" | wc -l) files to convert:"
echo "$v_files"
echo ""

# Rename each file
for file in $v_files; do
    new_file="${file%.v}.sv"
    echo "  $file → $new_file"
    git mv "$file" "$new_file"
done

echo ""
echo "✅ Conversion complete!"
echo "Next: Update build scripts to use .sv extension"
