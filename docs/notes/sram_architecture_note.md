# SRAM Architecture Note — CPU SRAM vs Tile SRAMs

## The Distinction

There are two completely separate categories of SRAM in this design:

### 1. CPU SRAM (built in Month 1)
- File: `hw/rtl/memory/sram_ctrl.sv`
- Size: 32KB
- Address: `SRAM_BASE = 0x1000_0000` in `soc_pkg.sv`
- Access: Via AXI crossbar — Ibex talks to it directly
- Purpose: Firmware stack, local variables, globals
- Location: Adjacent to AXI crossbar, outside the NoC mesh

### 2. Tile SRAMs (Month 5+)
- File: Not yet built
- Size: 16KB each, 3 instances (per physical spec)
- Address: NOT in soc_pkg — accessed via NoC packets, not AXI
- Purpose: Distributed storage nodes in the mesh (R(0,3), R(1,3), R(2,2))
- Location: Inside the NoC mesh as dedicated storage tiles

### 3. Output Scratchpads (Month 5+)
- File: Not yet built
- Size: 2KB per accelerator tile, 8 instances
- Access: Internal to each accelerator tile
- Purpose: Holds systolic array results before NoC forwarding
- Location: Inside each accelerator tile

## Key Rule

`SRAM_BASE` in `soc_pkg.sv` is **only** the CPU SRAM. Tile SRAMs do not get
entries in `soc_pkg` — they are addressed via NoC routing, not the AXI
address map.

## Action Required (Month 5)

When building tile SRAMs:
- Create separate controller modules (not sram_ctrl.sv)
- Do NOT add them to soc_pkg address map
- Wire them to NoC network interface, not AXI crossbar
- Size them at 16KB per the physical spec (not 32KB)

## Flag for Physical Collaborator

The physical spec does not explicitly list the CPU SRAM (32KB) in its area
estimates. It lists only the 3x tile SRAMs at 16KB each. The physical design
engineer needs to add one 32KB SRAM macro to the floorplan:
- Placement: Adjacent to AXI crossbar, outside NoC mesh
- Estimated area: ~600um x 800um at GF 130nm
- Generated via: OpenRAM or GF memory compiler at synthesis handoff
