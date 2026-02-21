#!/usr/bin/env python3
"""Strip SVA assertions and unsupported SV constructs for Yosys synthesis."""
import re, sys, os

src_dir = sys.argv[1]
dst_dir = sys.argv[2]

for root, dirs, files in os.walk(src_dir):
    if 'deprecated' in root:
        continue
    for fname in files:
        if not fname.endswith('.sv'):
            continue
        src = os.path.join(root, fname)
        rel = os.path.relpath(src, '.')
        dst = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        with open(src) as f:
            txt = f.read()

        # Remove property...endproperty blocks
        txt = re.sub(r'\bproperty\b\s+\w+.*?endproperty', '', txt, flags=re.DOTALL)

        # Remove assert/cover/assume property lines
        txt = re.sub(r'^\s*(assert|cover|assume)\s+property\b[^;]*;[^\n]*', '', txt, flags=re.MULTILINE)

        lines = txt.split('\n')
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip initial begin blocks that only have asserts
            if re.match(r'^\s*initial\s+begin\s*$', line):
                j = i + 1
                only_assert = True
                while j < len(lines):
                    s = lines[j].strip()
                    if s == 'end':
                        break
                    if s and not re.match(r'^(assert|//|\$|else)', s):
                        only_assert = False
                    j += 1
                if only_assert and j < len(lines) and lines[j].strip() == 'end':
                    i = j + 1
                    continue

            # Skip always blocks that only have asserts
            if re.match(r'^\s*always(_ff|_comb)?\s+(@\s*\(.*?\)\s*)?begin\s*$', line):
                j = i + 1
                only_assert = True
                while j < len(lines):
                    s = lines[j].strip()
                    if s == 'end':
                        break
                    if s and not re.match(r'^(assert|//|else\s+\$)', s):
                        only_assert = False
                    j += 1
                if only_assert and j < len(lines) and lines[j].strip() == 'end':
                    i = j + 1
                    continue

            # Remove standalone assert(...) ... ; (possibly multi-line)
            if re.match(r'^\s*assert\s*\(', stripped):
                combined = stripped
                j = i
                while ';' not in combined and j < len(lines) - 1:
                    j += 1
                    combined += ' ' + lines[j].strip()
                i = j + 1
                continue

            # Remove else $fatal/$error/$warning lines (orphaned from assert removal)
            if re.match(r'^\s*else\s+\$(fatal|error|warning)', stripped):
                i += 1
                continue

            out.append(line)
            i += 1

        with open(dst, 'w') as f:
            f.write('\n'.join(out))
        print(f"  {rel}: {len(lines)} -> {len(out)} lines")
