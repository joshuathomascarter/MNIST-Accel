#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PREP_DIR=".yosys_soc_top_v2"
SRC_DIR="$PREP_DIR/src"
SCRIPT_FILE="$PREP_DIR/soc_top_v2.ys"
LOG_FILE="reports/yosys_soc_top_v2.log"
JSON_FILE="reports/yosys_soc_top_v2.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Preparing soc_top_v2 sources for generic Yosys synthesis..."
rm -rf "$PREP_DIR"
mkdir -p "$SRC_DIR" reports

"$PYTHON_BIN" strip_assert.py rtl "$PREP_DIR/all" >/dev/null

while IFS= read -r raw_line; do
  line="${raw_line#${raw_line%%[![:space:]]*}}"
  [[ -z "$line" || "$line" == //* ]] && continue
  if [[ "$line" == *.sv ]]; then
    rel="${line#*/hw/}"
    [[ "$rel" == "rtl/top/soc_top.sv" ]] && continue
        [[ "$rel" == "rtl/memory/coherence_demo_top.sv" ]] && continue
        [[ "$rel" == "rtl/memory/directory_controller.sv" ]] && continue
        [[ "$rel" == "rtl/memory/snoop_filter.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_bandwidth_steal.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_innet_reduce.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_qos_shaper.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_router_sva.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_traffic_gen.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_vc_allocator_sparse.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_vc_allocator_qvn.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_vc_allocator_static_prio.sv" ]] && continue
        [[ "$rel" == "rtl/noc/noc_vc_allocator_weighted_rr.sv" ]] && continue
        [[ "$rel" == "rtl/noc/reduce_engine.sv" ]] && continue
        [[ "$rel" == "rtl/noc/scatter_engine.sv" ]] && continue
    mkdir -p "$SRC_DIR/$(dirname "$rel")"
    cp "$PREP_DIR/all/$rel" "$SRC_DIR/$rel"
  fi
done < sim/sv/filelist.f

"$PYTHON_BIN" - <<'PYEOF'
from pathlib import Path

root = Path('.')
prep_root = root / '.yosys_soc_top_v2' / 'src'
filelist = root / 'sim' / 'sv' / 'filelist.f'
script_path = root / '.yosys_soc_top_v2' / 'soc_top_v2.ys'
json_path = root / 'reports' / 'yosys_soc_top_v2.json'

include_dirs = []
files = []
for raw in filelist.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith('//'):
        continue
    if line.startswith('+incdir+'):
        rel = line[len('+incdir+'):].split('/hw/', 1)[1]
        include_dirs.append(rel)
    elif line.endswith('.sv'):
        rel = line.split('/hw/', 1)[1]
        if rel in {
            'rtl/top/soc_top.sv',
            'rtl/memory/coherence_demo_top.sv',
            'rtl/memory/directory_controller.sv',
            'rtl/memory/snoop_filter.sv',
            'rtl/noc/noc_bandwidth_steal.sv',
            'rtl/noc/noc_innet_reduce.sv',
            'rtl/noc/noc_qos_shaper.sv',
            'rtl/noc/noc_router_sva.sv',
            'rtl/noc/noc_traffic_gen.sv',
            'rtl/noc/noc_vc_allocator_sparse.sv',
            'rtl/noc/noc_vc_allocator_qvn.sv',
            'rtl/noc/noc_vc_allocator_static_prio.sv',
            'rtl/noc/noc_vc_allocator_weighted_rr.sv',
            'rtl/noc/reduce_engine.sv',
            'rtl/noc/scatter_engine.sv',
        }:
            continue
        files.append(rel)

include_flags = ' '.join(f'-I {prep_root / rel}' for rel in include_dirs)
script_lines = []
for rel in files:
    script_lines.append(
        f'read_verilog -sv -DSYNTHESIS -DASIC_SYNTHESIS {include_flags} {prep_root / rel}'
    )

script_lines += [
    'hierarchy -check -top soc_top_v2',
    'proc',
    'flatten',
    'opt -full',
    'synth -top soc_top_v2',
    'check',
    'stat',
    f'write_json {json_path}',
]

script_path.write_text('\n'.join(script_lines) + '\n')
print(f'Generated {script_path} with {len(files)} source files.')
PYEOF

echo "Running generic Yosys synthesis for soc_top_v2..."
yosys -s "$SCRIPT_FILE" 2>&1 | tee "$LOG_FILE"

echo "Done."
echo "  Log:  $LOG_FILE"
echo "  JSON: $JSON_FILE"
