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

# --- sv2v pass: convert all SV files to Verilog-2005 in-place ---
# This handles unpacked array ports and other SV constructs that Yosys
# doesn't natively support (e.g. logic [N-1:0] arr [M]).
echo "Running sv2v to flatten SV to Verilog-2005..."
INCDIR_FLAGS=()
while IFS= read -r raw_line; do
  line="${raw_line#${raw_line%%[![:space:]]*}}"
  [[ -z "$line" || "$line" == //* ]] && continue
  if [[ "$line" == +incdir+* ]]; then
    rel="${line#*+incdir+*/hw/}"
    INCDIR_FLAGS+=("--incdir=${SRC_DIR}/${rel}")
  fi
done < sim/sv/filelist.f

# --- Pass 1: per-file sv2v (handles files without package imports) ---
SED_FILTER='s/[$]test[$]plusargs([^)]*)/0/g; s/.*[$]display.*$/;/; s/.*[$]write.*$/;/; s/.*[$]monitor.*$/;/; s/.*[$]strobe.*$/;/; s/.*[$]fopen.*$/;/; s/.*[$]fclose.*$/;/; s/.*[$]time.*$/;/; s/.*[$]realtime.*$/;/; s/.*[$]finish.*$/;/; s/.*[$]stop.*$/;/; s/.*[$]fatal.*$/;/; s/.*[$]error.*$/;/'

find "$SRC_DIR" -name "*.sv" | while read -r svfile; do
  vfile="${svfile%.sv}.v"
  sv2v "${INCDIR_FLAGS[@]}" -DSYNTHESIS -DASIC_SYNTHESIS \
    --exclude=Assert --exclude=UnbasedUnsized \
    "$svfile" \
    | sed "$SED_FILTER" \
    > "$vfile" \
    && rm "$svfile" \
    || { echo "sv2v pass1 failed on $svfile (has package dep), will retry"; rm -f "$vfile"; }
done

# --- Pass 2: batch sv2v for files that still import packages (noc_pkg etc.) ---
# Pass noc_pkg.v from src dir (already converted) as the package source via
# the original .sv from the all/ snapshot, which sv2v can resolve.
NOC_PKG_SV="$PREP_DIR/all/rtl/noc/noc_pkg.sv"
SOC_PKG_SV="$PREP_DIR/all/rtl/top/soc_pkg.sv"

# Convert remaining .sv files, one at a time but with package files prepended
pkg_files=()
[[ -f "$NOC_PKG_SV" ]] && pkg_files+=("$NOC_PKG_SV")
[[ -f "$SOC_PKG_SV" ]] && pkg_files+=("$SOC_PKG_SV")

# Pre-process: strip sv2v-unsupported constructs from remaining .sv files
# 1. Remove parameterised casts PARAM_W'(expr) → (expr)  [sv2v 0.0.13 bug]
# 2. Remove assert/assume/cover statements → ;             [else sv2v leaves dangling if-bodies]
"$PYTHON_BIN" - <<'PYCAST'
import re, pathlib

src = pathlib.Path('.yosys_soc_top_v2/src')
# Pattern 1: UPPERCASE_W'( → (
cast_pat = re.compile(r'\b[A-Z][A-Z0-9_]*\'(\()')

def strip_asserts(text):
    """Remove assert/assume/cover statements, handling nested parens."""
    result = []
    i = 0
    n = len(text)
    kw_re = re.compile(r'\b(assert|assume|cover)\s*(?:#\([^)]*\)\s*)?', re.DOTALL)
    while i < n:
        m = kw_re.search(text, i)
        if not m:
            result.append(text[i:])
            break
        result.append(text[i:m.start()])
        j = m.end()
        if j >= n or text[j] != '(':
            result.append(text[m.start():j])
            i = j
            continue
        # skip balanced parens of the condition
        depth = 0
        while j < n:
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        # skip optional else clause until semicolon
        rest = text[j:]
        else_m = re.match(r'\s*else\s+[^;]*;', rest, re.DOTALL)
        if else_m:
            j += else_m.end()
        else:
            # skip trailing semicolon if present
            semi_m = re.match(r'\s*;', rest)
            if semi_m:
                j += semi_m.end()
        result.append(';')
        i = j
    return ''.join(result)

for sv in src.rglob('*.sv'):
    text  = sv.read_text(encoding='utf-8')
    fixed = cast_pat.sub(r'(', text)
    fixed = strip_asserts(fixed)
    if fixed != text:
        sv.write_text(fixed, encoding='utf-8')
        print(f'  preprocessed {sv}')
PYCAST

find "$SRC_DIR" -name "*.sv" | while read -r svfile; do
  vfile="${svfile%.sv}.v"
  echo "sv2v pass2 (with packages): $svfile"
  sv2v "${INCDIR_FLAGS[@]}" -DSYNTHESIS -DASIC_SYNTHESIS \
    --exclude=Assert --exclude=UnbasedUnsized \
    "${pkg_files[@]}" "$svfile" \
    | sed "$SED_FILTER" \
    > "$vfile" \
    && rm "$svfile" \
    || { echo "sv2v pass2 also failed on $svfile, keeping .sv"; rm -f "$vfile"; }
done
echo "sv2v conversion done."

# --- Stub out dram_phy_simple_mem: the 524K-word array OOMs Yosys ---
STUB_FILE="$SRC_DIR/rtl/dram/dram_phy_simple_mem.v"
cat > "$STUB_FILE" <<'STUB'
// Synthesis stub — replaces the 524K-word DRAM backing store.
// The memory is not synthesized; use BRAM inference in real
// implementation. This stub gives Yosys a proper black-box.
module dram_phy_simple_mem #(
    parameter NUM_BANKS  = 8,
    parameter ROW_BITS   = 14,
    parameter COL_BITS   = 10,
    parameter DATA_W     = 32,
    parameter MEM_WORDS  = 524288,
    parameter INIT_FILE  = ""
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [NUM_BANKS-1:0]  dram_phy_act,
    input  wire [NUM_BANKS-1:0]  dram_phy_read,
    input  wire [NUM_BANKS-1:0]  dram_phy_write,
    input  wire [NUM_BANKS-1:0]  dram_phy_pre,
    input  wire [ROW_BITS-1:0]   dram_phy_row,
    input  wire [COL_BITS-1:0]   dram_phy_col,
    input  wire                  dram_phy_ref,
    input  wire [DATA_W-1:0]     dram_phy_wdata,
    input  wire [DATA_W/8-1:0]   dram_phy_wstrb,
    output reg  [DATA_W-1:0]     dram_phy_rdata,
    output reg                   dram_phy_rdata_valid
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dram_phy_rdata       <= {DATA_W{1'b0}};
            dram_phy_rdata_valid <= 1'b0;
        end else begin
            dram_phy_rdata_valid <= |dram_phy_read;
            dram_phy_rdata       <= {DATA_W{1'b0}};
        end
    end
endmodule
STUB
echo "Stubbed dram_phy_simple_mem."

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
    # sv2v converted .sv -> .v; fall back to .sv if conversion kept original
    vrel = rel[:-3] + '.v'
    vpath = prep_root / vrel
    svpath = prep_root / rel
    src_path = vpath if vpath.exists() else svpath
    script_lines.append(
        f'read_verilog -sv -DSYNTHESIS -DASIC_SYNTHESIS {include_flags} {src_path}'
    )

script_lines += [
    'hierarchy -check -top soc_top_v2',
    'proc',
    'flatten',
    'opt -fast',          # fast pre-flatten cleanup
    # Structural elaboration only — no techmap/opt_reduce.
    # techmap is intentionally skipped: it triggers an exponential opt_expr loop
    # on this design size. OpenROAD does its own technology mapping with the PDK.
    # Goal here is: hierarchy OK + stat cell count + JSON netlist for handoff.
    'wreduce',
    'alumacc',
    'memory -nomap',
    'opt -fast',
    'fsm',
    'clean',
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
