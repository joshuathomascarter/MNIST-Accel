#!/usr/bin/env python3
"""
strip_sim.py — Remove simulation-only constructs from OPENROADMARCUS RTL files.

Handles:
  1. initial begin...end blocks containing $display, $readmemh, assert/$fatal,
     or for-loop array initialisation (e.g. sram[i] = '0) — deleted entirely
  2. Single-line initial $display(...); statements
  3. if ($test$plusargs(...)) blocks (single-line body or begin...end)
  4. Standalone $display / $write / $fwrite lines
  5. if (...$time...) lines (with their single-line body)
  6. always blocks whose only contents are $display / sim tasks

Run from anywhere:
  python3 strip_sim.py [directory]          # default: rtl/
"""

import re
import sys
import os
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

SIM_LINE_RE = re.compile(
    r'^\s*(\$display|\$write|\$fwrite|\$monitor|\$strobe|\$readmemh|\$readmemb'
    r'|\$dumpfile|\$dumpvars|\$finish|\$stop)\s*\(',
    re.IGNORECASE
)
FATAL_LINE_RE = re.compile(r'^\s*(else\s+)?\$fatal\b')
PLUSARGS_RE   = re.compile(r'\$test\$plusargs\s*\(')
TIME_IN_IF_RE = re.compile(r'^\s*if\s*\(.*\$time\b')


def is_sim_only_line(line):
    """True if this line is purely a simulation statement."""
    s = line.strip()
    if not s or s.startswith('//'):
        return False
    return bool(SIM_LINE_RE.match(line) or FATAL_LINE_RE.match(line))


def collect_block(lines, start, keyword='begin', end_kw='end'):
    """
    Starting at lines[start] which contains `keyword`, collect until matching
    `end_kw`.  Returns (end_index_inclusive, body_lines).
    """
    depth = 0
    i = start
    body = []
    found = False
    while i < len(lines):
        l = lines[i]
        stripped = l.strip()
        # count begin/end only in non-comment tokens
        no_comment = re.sub(r'//.*', '', l)
        opens  = len(re.findall(r'\b(begin|fork)\b', no_comment))
        closes = len(re.findall(r'\b(end|join)\b',   no_comment))
        if not found:
            if keyword in no_comment:
                found = True
                depth += opens - closes
        else:
            depth += opens - closes
            body.append(l)
        if found and depth <= 0:
            return i, body
        i += 1
    return i - 1, body


def initial_block_is_sim_only(body_lines):
    """Return True if every meaningful line in a begin...end body is sim-only."""
    for l in body_lines:
        s = l.strip()
        if not s or s.startswith('//') or s in ('begin', 'end', 'end;'):
            continue
        # for-loop that only assigns to array (e.g. sram[i] = '0) → sim/init only
        if re.match(r'for\s*\(', s):
            continue
        if re.match(r'\w+\s*\[.*\]\s*=', s):
            continue
        if is_sim_only_line(l):
            continue
        if re.match(r'assert\s*\(', s):
            continue
        if re.match(r'else\s*\$', s):
            continue
        if re.match(r'if\s*\(.*\)\s*$', s):   # bare if (no body on same line)
            continue
        if re.match(r'if\s*\(.*\)\s+begin\s*$', s):  # if (...) begin with empty body
            continue
        # Anything else → block has real logic
        return False
    return True


def strip_file(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out   = []
    i     = 0
    while i < len(lines):
        line = lines[i]
        raw  = line.rstrip('\n\r')

        # ── 0. `ifdef SIMULATION ... `endif blocks ───────────────────────────
        if re.match(r'\s*`ifdef\s+SIMULATION\b', raw):
            depth = 1
            i += 1
            while i < len(lines) and depth > 0:
                l2 = lines[i].rstrip('\n\r')
                if re.match(r'\s*`ifdef\b', l2):
                    depth += 1
                elif re.match(r'\s*`endif\b', l2):
                    depth -= 1
                i += 1
            continue

        stripped = raw.strip()
        no_comment = re.sub(r'//.*', '', raw)

        # ── 1. initial begin...end (multi-line) ──────────────────────────────
        if re.match(r'\s*initial\s+begin\b', raw):
            end_i, body = collect_block(lines, i)
            if initial_block_is_sim_only(body):
                i = end_i + 1
                continue
            # keep as-is
            out.append(line)
            i += 1
            continue

        # ── 2. initial <single statement>; ───────────────────────────────────
        if re.match(r'\s*initial\s+\$', raw):
            i += 1
            continue

        # ── 3. if ($test$plusargs(...)) ──────────────────────────────────────
        if PLUSARGS_RE.search(no_comment) and re.match(r'\s*if\s*\(', raw):
            # Consume the if line itself
            i += 1
            # Check if next non-blank is a begin block or single statement
            while i < len(lines) and not lines[i].strip():
                i += 1  # skip blank lines
            if i < len(lines):
                nxt = lines[i].strip()
                if nxt.startswith('begin') or (re.search(r'\bbegin\b', lines[i]) and
                        not re.search(r'\bend\b', lines[i])):
                    end_i, _ = collect_block(lines, i)
                    i = end_i + 1
                else:
                    # single-line body
                    i += 1
            continue

        # ── 4. Standalone $display / $write etc. (possibly multi-line) ───────
        if SIM_LINE_RE.match(line):
            # consume until semicolon
            combined = raw
            while ';' not in combined and i + 1 < len(lines):
                i += 1
                combined += lines[i].rstrip('\n\r')
            i += 1
            continue

        # ── 5. else $fatal lines ─────────────────────────────────────────────
        if FATAL_LINE_RE.match(line):
            i += 1
            continue

        # ── 6. if (...$time...) single-line body ─────────────────────────────
        if TIME_IN_IF_RE.match(raw):
            i += 1
            # consume single-line body if next line is $display
            if i < len(lines) and SIM_LINE_RE.match(lines[i]):
                combined = lines[i].rstrip('\n\r')
                while ';' not in combined and i + 1 < len(lines):
                    i += 1
                    combined += lines[i].rstrip('\n\r')
                i += 1
            continue

        out.append(line)
        i += 1

    return ''.join(out)


def process_dir(root: Path):
    sv_files = list(root.rglob('*.sv')) + list(root.rglob('*.v'))
    changed = 0
    for f in sv_files:
        original = f.read_text(encoding='utf-8', errors='replace')
        cleaned  = strip_file(original)
        if cleaned != original:
            f.write_text(cleaned, encoding='utf-8')
            print(f"  stripped: {f.relative_to(root.parent)}")
            changed += 1
        else:
            print(f"  ok:       {f.relative_to(root.parent)}")
    print(f"\nDone — {changed}/{len(sv_files)} files modified.")


if __name__ == '__main__':
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'rtl'
    if not target.exists():
        print(f"ERROR: {target} does not exist")
        sys.exit(1)
    print(f"Stripping sim constructs from: {target}\n")
    process_dir(target)
