# RTL Reverse-Engineering And Rewrite Study Plan

This plan is for learning the current larger ACCEL-v1 project as an RTL engineer, not just reading it passively. The goal is to reverse-engineer the architecture, redraw it in draw.io, rewrite the important modules yourself, then compare your version against the repository.

My view: this is a very strong way to learn. The only change I would make to your original idea is this: do not force a rigid "one file every single day" rule. Some files are half-day wrappers. Some files need two or three days. The right unit is not "one file". The right unit is "one concept boundary".

This roadmap therefore does three things:

1. It keeps the six-hour day structure you want.
2. It covers the current bigger SoC-scale RTL, not only the old small accelerator.
3. It deliberately excludes the original small-accelerator baseline from the mandatory rewrite track, because you said you already understand that path and want to move into caches, scratchpads, NoC, DRAM, and full-system design.

## How To Use This Plan

- Rewrite first from understanding, not by copying.
- Only compare against repo RTL after your first pass is done.
- Keep one draw.io file with separate pages for: full system, memory hierarchy, cache flow, NoC flow, tile flow, and software-to-hardware control flow.
- Keep one markdown notebook for mistakes, questions, and "what this module is really doing" summaries.
- If a day says "minimal prelearning", spend essentially the full six hours coding, testing, and drawing.

## Default Six-Hour Day Template

Use this unless the day entry overrides it.

- 45 minutes: prelearning or rereading yesterday's notes
- 3 hours 45 minutes: rewrite the target module or module cluster
- 45 minutes: compare your rewrite to the repo version
- 45 minutes: update draw.io and write a one-page summary

For heavy theory days, use this instead:

- 90 minutes: prelearning
- 3 hours 30 minutes: rewrite
- 60 minutes: compare and document

For pure implementation days, use this instead:

- 6 hours: coding, simulation, waveform inspection, and notes

## Mandatory Scope

This plan covers the current larger RTL system under `hw/rtl/`, especially:

- top-level SoC integration
- CPU, boot, translation, and OBI-to-AXI bridging
- L1 and L2 caches
- DRAM front-end and scheduling
- scratchpad, tile controller, DMA gateway, and tile integration
- full NoC routing, allocation, transport, and reduction plumbing
- peripherals, interrupts, and performance monitoring

## Excluded From The Mandatory Rewrite Track

Treat these as reference-only baseline files because they belong to the older small-accelerator flow you already understand:

- `hw/rtl/top/accel_top.sv`
- `hw/rtl/buffer/act_buffer.sv`
- `hw/rtl/buffer/max_pool_2x2.sv`
- `hw/rtl/buffer/output_accumulator.sv`
- `hw/rtl/buffer/output_bram_buffer.sv`
- `hw/rtl/buffer/output_bram_ctrl.sv`
- `hw/rtl/buffer/wgt_buffer.sv`
- `hw/rtl/control/bsr_scheduler.sv`
- `hw/rtl/control/csr.sv`
- `hw/rtl/dma/act_dma.sv`
- `hw/rtl/dma/bsr_dma.sv`
- `hw/rtl/dma/dma_pack_112.sv`
- `hw/rtl/dma/out_dma.sv`
- `hw/rtl/host_iface/axi_dma_bridge.sv`
- `hw/rtl/host_iface/axi_lite_slave.sv`
- `hw/rtl/mac/mac8.sv`
- `hw/rtl/monitor/perf.sv`
- `hw/rtl/systolic/pe.sv`
- `hw/rtl/systolic/systolic_array_sparse.sv`

If later you want a second roadmap just for the legacy baseline stack, make that a separate appendix rather than mixing it into this main plan.

## Companion Context Files To Read Alongside The RTL

These are not mandatory rewrite targets in this roadmap, but they are important because they show how software, firmware, and memory-mapped control meet the RTL:

- `fw/main.c`
- `fw/hal_accel.h`
- `sw/ml_python/host/accel.py`
- `sw/ml_python/host/axi_driver.py`
- `sw/ml_python/host/memory.py`
- `sw/cpp/include/memory/address_map.hpp`
- `sw/cpp/include/memory/dma_controller.hpp`
- `sw/cpp/include/memory/buffer_manager.hpp`
- `sw/cpp/include/driver/accelerator.hpp`
- `sw/cpp/include/driver/csr_interface.hpp`
- `sw/cpp/include/driver/performance.hpp`

Read these on the days where the matching RTL path is being studied.

## The Core Learning Loop

For each day, do the same four-step loop.

1. Draw the block before you rewrite it.
2. Rewrite the block from behavior, interfaces, and invariants.
3. Compare against the repository version and mark what you missed.
4. Write one short paragraph on why this block exists architecturally.

---

## Week 1: System Framing And CPU Entry Path

### Day 1: Map The Whole System Before Touching RTL

- Files: companion context only: `README.md`, `docs/architecture/ARCHITECTURE.md`, `fw/main.c`, `sw/ml_python/host/accel.py`, `sw/cpp/include/memory/address_map.hpp`
- Prelearning: 60 to 90 minutes on memory-mapped I/O, AXI4-Lite vs AXI4, master/slave terminology, clocks and resets
- Work: build your first draw.io pages for full system architecture and file architecture; list every top-level master, slave, memory region, interrupt, and clock domain you can find
- Why this matters: if you skip this, later cache and NoC files will feel disconnected and random

### Day 2: Read The Real Top Of The New System

- Files: `hw/rtl/top/soc_pkg.sv`, `hw/rtl/top/soc_top_v2.sv`
- Prelearning: 45 minutes on SoC address maps, top-level integration style, and parameter packages
- Work: mark every submodule instantiation in `soc_top_v2.sv`; identify the CPU path, cache path, tile-array path, peripheral path, and DRAM path; rewrite `soc_pkg.sv` from scratch and produce a one-page `soc_top_v2` block diagram
- Why this matters: `soc_top_v2.sv` is the architectural truth of the current design

### Day 3: Learn The Delta Between The Older SoC Top And The New One

- Files: `hw/rtl/top/soc_top.sv`
- Prelearning: minimal; spend nearly the full day coding and diffing
- Work: reverse-engineer how `soc_top.sv` differs from `soc_top_v2.sv`; write a "why v2 exists" note; redraw only the changed paths
- Why this matters: learning design evolution teaches you how architects grow a system without losing the old intent

### Day 4: Rewrite The CPU Control Core, Part 1

- Files: `hw/rtl/top/simple_cpu.sv`
- Prelearning: 90 minutes on RV32I fetch/decode/execute state machines, register files, and simple CSR handling
- Work: rewrite fetch, decode, register file, ALU operations, branch logic, and state transitions through execute
- Why this matters: the CPU is the first active master in the SoC and explains how software enters the hardware world

### Day 5: Rewrite The CPU Control Core, Part 2, And Boot Path

- Files: `hw/rtl/top/simple_cpu.sv`, `hw/rtl/memory/boot_rom.sv`
- Prelearning: 60 minutes on load/store alignment, trap return, WFI, and boot sequencing
- Work: finish the memory-side state machine, CSR semantics, and boot behavior; rewrite `boot_rom.sv`; then draw the reset-to-first-fetch sequence
- Why this matters: this is where bare-metal firmware becomes bus transactions

### Day 6: Bridge OBI To AXI

- Files: `hw/rtl/top/obi_to_axi.sv`
- Prelearning: 60 minutes on ready/valid handshakes, request/response decoupling, and protocol adaptation
- Work: rewrite the bridge, paying attention to ordering, request lifetime, and how simple CPU semantics are converted into AXI-style requests
- Why this matters: this module is the first real protocol boundary in the new system

### Day 7: Decode And Arbitrate The Fabric

- Files: `hw/rtl/top/axi_addr_decoder.sv`, `hw/rtl/top/axi_arbiter.sv`
- Prelearning: 45 minutes on memory maps, interconnect arbitration, and backpressure
- Work: rewrite both files and draw the path from one initiator to multiple targets; note what information is purely routing and what is control policy
- Why this matters: this is the minimum logic that turns a point-to-point bus into a system fabric

---

## Week 2: Crossbar, SRAM Path, Translation, And Peripherals

### Day 8: Rewrite The Crossbar, Part 1

- Files: `hw/rtl/top/axi_crossbar.sv`
- Prelearning: 60 minutes on multi-master AXI routing, outstanding transactions, and channel independence
- Work: rewrite the address-routing half of the crossbar; identify read and write channel structure separately instead of treating AXI as one bus
- Why this matters: the crossbar is where architectural neatness usually breaks if you do not think clearly about independent channels

### Day 9: Rewrite The Crossbar, Part 2

- Files: `hw/rtl/top/axi_crossbar.sv`
- Prelearning: minimal; make this a full coding day
- Work: finish write-data, read-data, and response routing; compare your version to the repo and explicitly list what had to be buffered or staged to avoid combinational problems
- Why this matters: large integration files teach real engineering judgment, not just syntax

### Day 10: Learn The Simple On-Chip Memory Path

- Files: `hw/rtl/memory/sram_1rw_wrapper.sv`, `hw/rtl/memory/sram_ctrl.sv`
- Prelearning: 45 minutes on single-port SRAM timing, byte enables, and memory wrappers
- Work: rewrite both files; draw how an abstract SRAM macro is wrapped and then controlled at the SoC level
- Why this matters: before tackling caches and DRAM, you want a clean mental model for the simplest memory block in the project

### Day 11: Rewrite The TLB, Part 1

- Files: `hw/rtl/memory/tlb.sv`
- Prelearning: 90 minutes on VPN/PPN, hit/miss lookup, permission bits, and Sv32 basics
- Work: rewrite the storage structure, lookup path, and fill/update idea for the TLB
- Why this matters: the TLB is where virtual memory stops being abstract and becomes exact hardware state

### Day 12: Finish Translation With The Page Table Walker

- Files: `hw/rtl/memory/tlb.sv`, `hw/rtl/memory/page_table_walker.sv`
- Prelearning: 60 minutes on two-level page walks, leaf vs non-leaf PTEs, and permission faults
- Work: finish TLB control, rewrite the PTW FSM, and draw the miss-to-fill path from TLB through memory and back
- Why this matters: this teaches how control FSMs interact with memory protocol and protection logic

### Day 13: Learn UART End To End

- Files: `hw/rtl/periph/uart_tx.sv`, `hw/rtl/periph/uart_rx.sv`, `hw/rtl/periph/uart_ctrl.sv`
- Prelearning: 45 minutes on UART framing, baud timing, and memory-mapped peripheral design
- Work: rewrite the transmitter, receiver, and controller register layer; compare them to firmware use in `fw/main.c`
- Why this matters: simple peripherals are the cleanest place to learn register-interface discipline

### Day 14: Learn Timers, GPIO, And Interrupt Collection

- Files: `hw/rtl/periph/timer_ctrl.sv`, `hw/rtl/periph/gpio_ctrl.sv`, `hw/rtl/periph/plic.sv`
- Prelearning: 60 minutes on periodic timers, GPIO direction registers, and interrupt prioritization
- Work: rewrite all three, then draw how external events become CPU-visible interrupts
- Why this matters: these blocks make the SoC feel like a real computer, not only an accelerator demo

---

## Week 3: L1 And L2 Cache Architecture

### Day 15: Rewrite The L1 Data And Tag Arrays

- Files: `hw/rtl/cache/l1_data_array.sv`, `hw/rtl/cache/l1_tag_array.sv`
- Prelearning: 60 minutes on set-associative caches, tags, index/offset decomposition, and dirty bits
- Work: rewrite both arrays and draw the lookup fields from an address bit-slice perspective
- Why this matters: arrays are the physical substrate of the cache; if you understand them, controllers become much easier

### Day 16: Learn LRU And L1 Control, Part 1

- Files: `hw/rtl/cache/l1_lru.sv`, `hw/rtl/cache/l1_cache_ctrl.sv`
- Prelearning: 60 minutes on replacement policy, write-back vs write-through, and write-allocate behavior
- Work: rewrite LRU and the first half of the L1 controller, especially hit/miss classification and state sequencing
- Why this matters: this is the first place where policy meets storage arrays

### Day 17: Finish L1 Control And Start The L1 Top

- Files: `hw/rtl/cache/l1_cache_ctrl.sv`, `hw/rtl/cache/l1_dcache_top.sv`
- Prelearning: minimal; use a full implementation day
- Work: finish the controller and start the top wrapper around arrays and control; diagram the CPU request to cache response path
- Why this matters: top-level cache integration is where local module understanding becomes system understanding

### Day 18: Finish The L1 Data Cache Top

- Files: `hw/rtl/cache/l1_dcache_top.sv`
- Prelearning: 45 minutes on refill and eviction sequences
- Work: finish the L1 top and note exactly how it talks to downstream memory and upstream CPU logic
- Why this matters: this file is the practical boundary between processor-side latency and memory-side complexity

### Day 19: Rewrite The L2 Storage Arrays

- Files: `hw/rtl/cache/l2_data_array.sv`, `hw/rtl/cache/l2_tag_array.sv`
- Prelearning: 45 minutes on larger shared caches, wider lines, and tag/data decoupling
- Work: rewrite both arrays and compare what changed relative to L1 beyond just size
- Why this matters: it teaches the difference between a private near-core cache and a lower shared storage layer

### Day 20: Rewrite The L2 MSHR And Prefetcher

- Files: `hw/rtl/cache/l2_mshr.sv`, `hw/rtl/cache/stride_prefetcher.sv`
- Prelearning: 90 minutes on non-blocking caches, miss status holding registers, and stride prediction
- Work: rewrite both files and draw how outstanding misses are tracked independently of the request source
- Why this matters: this is one of the first advanced architecture concepts that feels like real microarchitecture instead of classroom cache diagrams

### Day 21: Rewrite The L2 Top, Part 1

- Files: `hw/rtl/cache/l2_cache_top.sv`
- Prelearning: 60 minutes on shared-cache request arbitration and refill/writeback flow
- Work: rewrite the upstream-facing half and the main controller skeleton
- Why this matters: `l2_cache_top.sv` is one of the main architecture files in the whole repo

---

## Week 4: Shared Cache Completion And DRAM Front-End

### Day 22: Rewrite The L2 Top, Part 2

- Files: `hw/rtl/cache/l2_cache_top.sv`
- Prelearning: minimal; full coding day
- Work: implement eviction, refill, and hazard handling in your rewrite; compare against the real file and note every place the design added staging or buffering
- Why this matters: hard files teach why clean architecture on paper still needs very practical timing-safe structure in RTL

### Day 23: Rewrite The L2 Top, Part 3, And Freeze Your Cache Diagram

- Files: `hw/rtl/cache/l2_cache_top.sv`
- Prelearning: none; use the full day for coding, diagramming, and self-review
- Work: finish the rewrite, produce a final L1/L2 hierarchy draw.io page, and write a two-page cache flow note
- Why this matters: before you move into DRAM, your cache mental model must be stable

### Day 24: Start The DRAM Front-End With Address, Queue, And Write Buffering

- Files: `hw/rtl/dram/dram_addr_decoder.sv`, `hw/rtl/dram/dram_cmd_queue.sv`, `hw/rtl/dram/dram_write_buffer.sv`
- Prelearning: 60 minutes on bank/row/column mapping, request queues, and decoupled write data buffering
- Work: rewrite all three; draw how one AXI address becomes bank/row/column and then becomes queued state
- Why this matters: DRAM controllers are impossible to understand if you do not first isolate the front-end bookkeeping

### Day 25: Rewrite The Bank FSM And Refresh Controller

- Files: `hw/rtl/dram/dram_bank_fsm.sv`, `hw/rtl/dram/dram_refresh_ctrl.sv`
- Prelearning: 90 minutes on ACT, READ, WRITE, PRE, tRCD, tRP, tRAS, and refresh timing basics
- Work: rewrite both files and draw one bank's legal state transitions
- Why this matters: every later scheduler decision is constrained by these exact timing states

### Day 26: Rewrite The DRAM Scheduler, Part 1

- Files: `hw/rtl/dram/dram_scheduler_frfcfs.sv`
- Prelearning: 90 minutes on FR-FCFS scheduling and row-buffer hits vs misses
- Work: rewrite the entry classification and candidate-selection logic first
- Why this matters: DRAM performance is largely a scheduling problem, not only a storage problem

### Day 27: Finish The DRAM Scheduler And Add Deterministic Mode

- Files: `hw/rtl/dram/dram_scheduler_frfcfs.sv`, `hw/rtl/dram/dram_deterministic_mode.sv`
- Prelearning: 45 minutes on deterministic service vs throughput-oriented service
- Work: finish the scheduler and then rewrite `dram_deterministic_mode.sv`; write down where the repo favors realism and where it favors controllability
- Why this matters: this is good architecture taste training because it exposes tradeoffs between performance and predictability

### Day 28: Power Modeling And DRAM Top Integration, Part 1

- Files: `hw/rtl/dram/dram_power_model.sv`, `hw/rtl/dram/dram_ctrl_top.sv`
- Prelearning: 45 minutes on DRAM command energy accounting and top-level controller integration
- Work: rewrite the power model and the first half of `dram_ctrl_top.sv`, especially how subblocks are composed
- Why this matters: top files are where separate concepts become a usable memory subsystem

---

## Week 5: DRAM Completion, Coherence Concepts, And Tile-Local Control

### Day 29: Finish The DRAM Controller Top

- Files: `hw/rtl/dram/dram_ctrl_top.sv`
- Prelearning: minimal; full implementation day
- Work: finish the rewrite, then draw the full AXI-to-DRAM path with queues, bank FSMs, scheduler, refresh, and PHY-side outputs
- Why this matters: after this day you should be able to explain the entire external-memory path in one diagram from memory

### Day 30: Learn The Coherence Package And Directory Controller

- Files: `hw/rtl/memory/coherence_pkg.sv`, `hw/rtl/memory/directory_controller.sv`
- Prelearning: 90 minutes on coherence states, sharer tracking, directory-based invalidation, and why many research designs expose packages for state definitions
- Work: rewrite the package and controller; do not worry yet about a complete industrial protocol, just understand the local machinery
- Why this matters: even partial coherence logic teaches how metadata and policy interact in multi-agent systems

### Day 31: Learn Snoop Filtering And The Coherence Demo Top

- Files: `hw/rtl/memory/snoop_filter.sv`, `hw/rtl/memory/coherence_demo_top.sv`
- Prelearning: 60 minutes on snoop reduction and why filters exist
- Work: rewrite both files and note what is experimental or demonstrative versus what is core production architecture
- Why this matters: learning to distinguish "core path" from "research/demo path" is an important engineering skill

### Day 32: Learn Clock Gating And Barrier Synchronization

- Files: `hw/rtl/control/clock_gate_cell.sv`, `hw/rtl/control/barrier_sync.sv`
- Prelearning: 60 minutes on integrated clock gates, enable-safe clock gating, and barrier release logic
- Work: rewrite both files and draw where clock control is allowed and where it is intentionally not used
- Why this matters: power-aware RTL is a real engineering topic, not a decorative extra

### Day 33: Rewrite The Tile Scratchpad

- Files: `hw/rtl/systolic/accel_scratchpad.sv`
- Prelearning: 45 minutes on scratchpad vs cache, banked memories, and explicit software-managed local storage
- Work: rewrite the scratchpad and compare it conceptually to the old activation and weight buffers you already know
- Why this matters: this is one of the key steps from "small accelerator" to "tile in a larger SoC"

### Day 34: Rewrite The Tile Controller, Part 1

- Files: `hw/rtl/control/tile_controller.sv`
- Prelearning: 60 minutes on command-driven accelerators, local FSM control, and DMA/compute/store sequencing
- Work: rewrite the opcode, state, and command interpretation portions first
- Why this matters: `tile_controller.sv` is the local brain of a compute tile

### Day 35: Rewrite The Tile Controller, Part 2

- Files: `hw/rtl/control/tile_controller.sv`
- Prelearning: minimal; full implementation day
- Work: finish DMA handshakes, barrier behavior, sparse hints, and done/error signaling; then draw the tile lifecycle from idle to done
- Why this matters: if you understand this file deeply, you understand how compute gets orchestrated at the tile level

---

## Week 6: NoC Fundamentals

### Day 36: Start The NoC With Shared Types And Routing Primitives

- Files: `hw/rtl/noc/noc_pkg.sv`, `hw/rtl/noc/noc_route_compute.sv`, `hw/rtl/noc/noc_credit_counter.sv`
- Prelearning: 90 minutes on wormhole flow control, flits, virtual channels, credits, and XY routing
- Work: rewrite these three first so all later NoC files have a stable vocabulary and mental model
- Why this matters: packages and tiny helpers define the semantics of the whole interconnect

### Day 37: Rewrite The NoC Input Port, Part 1

- Files: `hw/rtl/noc/noc_input_port.sv`
- Prelearning: 60 minutes on per-VC FIFOs, head-flit routing, and flow control boundaries
- Work: rewrite the FIFO and VC bookkeeping side first
- Why this matters: input ports are where transport theory turns into actual queued state and congestion behavior

### Day 38: Finish The Input Port And Rewrite The Crossbar

- Files: `hw/rtl/noc/noc_input_port.sv`, `hw/rtl/noc/noc_crossbar_5x5.sv`
- Prelearning: 45 minutes on switch fabrics and route selection
- Work: finish the input port and rewrite the local router crossbar
- Why this matters: the combination of input buffering plus a crossbar is the physical core of a router datapath

### Day 39: Rewrite The Switch Allocator

- Files: `hw/rtl/noc/noc_switch_allocator.sv`
- Prelearning: 90 minutes on iSLIP-style arbitration and round-robin fairness
- Work: rewrite the two-phase grant logic and then explain in your own words how one winning flit per output is chosen
- Why this matters: allocators are where router quality and correctness live or die

### Day 40: Rewrite The Baseline VC Allocator

- Files: `hw/rtl/noc/noc_vc_allocator.sv`
- Prelearning: 60 minutes on VC assignment, deadlock avoidance, and allocator policy vs transport correctness
- Work: rewrite the baseline allocator and note how its structure differs from the switch allocator
- Why this matters: this is the other half of router resource arbitration

### Day 41: Rewrite The Alternative VC Allocators

- Files: `hw/rtl/noc/noc_vc_allocator_static_prio.sv`, `hw/rtl/noc/noc_vc_allocator_weighted_rr.sv`, `hw/rtl/noc/noc_vc_allocator_sparse.sv`, `hw/rtl/noc/noc_vc_allocator_qvn.sv`
- Prelearning: 60 minutes on policy design: priority, weighted fairness, sparse-aware allocation, and QoS-oriented variants
- Work: rewrite all four and make a comparison table of what changes and what remains invariant
- Why this matters: architecture is often about interchangeable policy modules around a stable datapath shell

### Day 42: Rewrite The Router, Part 1

- Files: `hw/rtl/noc/noc_router.sv`
- Prelearning: 60 minutes on how input ports, allocators, and crossbars compose into a full router
- Work: rewrite the structural composition and main control/data plumbing first
- Why this matters: `noc_router.sv` is one of the major files in the project and ties many NoC concepts together

---

## Week 7: NoC System-Level Integration

### Day 43: Finish The Router And Build The Mesh

- Files: `hw/rtl/noc/noc_router.sv`, `hw/rtl/noc/noc_mesh_4x4.sv`
- Prelearning: 45 minutes on mesh topology, port conventions, and local vs inter-router links
- Work: finish the router and rewrite the mesh tiling around it; then draw the 4x4 link map
- Why this matters: this is where a single-router understanding becomes a network understanding

### Day 44: Rewrite The Network Interface, Part 1

- Files: `hw/rtl/noc/noc_network_interface.sv`
- Prelearning: 60 minutes on endpoint packetization, DMA request framing, and tile-to-network adaptation
- Work: rewrite the transmit-side logic and command/data packet formation first
- Why this matters: the NI is the bridge between local compute semantics and network transport semantics

### Day 45: Rewrite The Network Interface, Part 2

- Files: `hw/rtl/noc/noc_network_interface.sv`
- Prelearning: minimal; full coding day
- Work: finish receive-side handling, outstanding request state, DMA response behavior, and tile-facing status
- Why this matters: this file explains how the tile controller actually gets data into and out of the NoC world

### Day 46: Rewrite The Reduction And Scatter Primitives

- Files: `hw/rtl/noc/reduce_engine.sv`, `hw/rtl/noc/scatter_engine.sv`, `hw/rtl/noc/tile_reduce_consumer.sv`
- Prelearning: 60 minutes on reduction trees, scatter semantics, and root-side accumulation behavior
- Work: rewrite all three and draw message format expectations for reduce/scatter traffic
- Why this matters: this is where the NoC becomes computation-aware rather than only transport-aware

### Day 47: Rewrite In-Network Reduction Plumbing And Traffic Generation

- Files: `hw/rtl/noc/noc_innet_reduce.sv`, `hw/rtl/noc/noc_traffic_gen.sv`
- Prelearning: 60 minutes on synthetic traffic, in-network reduction, and experiment-only plumbing vs production transport
- Work: rewrite both and note which pieces are infrastructure for verification/experimentation versus normal runtime paths
- Why this matters: strong engineers learn to separate experimental scaffolding from mainline design intent

### Day 48: Rewrite QoS And Bandwidth-Shaping Modules

- Files: `hw/rtl/noc/qos_arbiter.sv`, `hw/rtl/noc/noc_qos_shaper.sv`, `hw/rtl/noc/noc_bandwidth_steal.sv`
- Prelearning: 60 minutes on traffic shaping, fairness, and service differentiation
- Work: rewrite the three policy modules and explain what problem each one solves that a plain mesh would not
- Why this matters: these files teach system-level tradeoffs rather than only local logic correctness

### Day 49: Rewrite The Tile DMA Gateway, Part 1

- Files: `hw/rtl/systolic/tile_dma_gateway.sv`
- Prelearning: 60 minutes on tile-local DMA front-ends, burst tracking, and gateway semantics between tiles and memory
- Work: rewrite the request side and local buffering or staging behavior first
- Why this matters: this file is one of the most important bridges between compute tiles, the NoC, and external memory

---

## Week 8: Tile Integration And Performance Observability

### Day 50: Finish The Tile DMA Gateway

- Files: `hw/rtl/systolic/tile_dma_gateway.sv`
- Prelearning: minimal; full implementation day
- Work: finish write/read completion behavior, outstanding transaction bookkeeping, and backpressure handling
- Why this matters: gateway bugs are usually integration bugs, so this file builds serious engineering instincts

### Day 51: Rewrite The Accelerator Tile, Part 1

- Files: `hw/rtl/systolic/accel_tile.sv`
- Prelearning: 60 minutes on local compute composition: controller + scratchpad + NI + gateway + systolic core
- Work: rewrite the structural shell and local interface map first
- Why this matters: `accel_tile.sv` is the cleanest file for learning how a tile is assembled from heterogeneous subblocks

### Day 52: Finish The Accelerator Tile

- Files: `hw/rtl/systolic/accel_tile.sv`
- Prelearning: minimal; full coding day
- Work: finish local handshakes, clocking choices, and completion signaling; then draw the entire tile dataflow from command to result
- Why this matters: once you can explain this file, the leap from block-level to subsystem-level design is mostly done

### Day 53: Rewrite The Tile Array, Part 1

- Files: `hw/rtl/systolic/accel_tile_array.sv`
- Prelearning: 60 minutes on replicated structures, mesh attachment, and top-level array control
- Work: rewrite the array shell, tile replication, and local port stitching first
- Why this matters: array-level design is where elegant local blocks must survive scale-out

### Day 54: Finish The Tile Array

- Files: `hw/rtl/systolic/accel_tile_array.sv`
- Prelearning: minimal; full implementation day
- Work: finish AXI-facing behavior, local CSR routing, barrier interactions, and NoC attachment
- Why this matters: this file is the point where the compute subsystem becomes a system-level citizen

### Day 55: Rewrite Performance Monitoring And Freeze The NoC/Tile Diagrams

- Files: `hw/rtl/monitor/perf_axi.sv`
- Prelearning: 45 minutes on performance counters, observability, and why metrics should be designed in, not bolted on
- Work: rewrite `perf_axi.sv`; finish your NoC and tile draw.io pages; add one page that shows where you would probe or instrument the system during debug
- Why this matters: good RTL engineers build for observability, not only functionality

### Day 56: Write The Full Memory And Transport Narrative

- Files: no new mandatory RTL; use the files from days 24 to 55
- Prelearning: none
- Work: spend the full day writing one integrated note that explains the path from CPU command to cache to DRAM to NoC to tile scratchpad and back
- Why this matters: forcing yourself to narrate the entire path reveals weak understanding immediately

---

## Week 9: Full-System Reintegration And Independent Design Practice

### Day 57: Rebuild `soc_top_v2` From The CPU And Memory Side

- Files: `hw/rtl/top/soc_top_v2.sv`
- Prelearning: minimal; this is a synthesis day for your understanding
- Work: redraw and re-annotate only the CPU, TLB/PTW, cache, SRAM, DRAM, and peripheral portions of `soc_top_v2.sv`
- Why this matters: reintegration is the real test of whether you learned separate files as one architecture

### Day 58: Rebuild `soc_top_v2` From The Tile And NoC Side

- Files: `hw/rtl/top/soc_top_v2.sv`
- Prelearning: none
- Work: redraw and re-annotate the tile-array, NoC, barrier, DMA gateway, and accelerator-facing parts of the top
- Why this matters: many people understand compute blocks and interconnect blocks separately but fail to join them correctly

### Day 59: Rebuild `soc_top_v2` From The Peripheral And Interrupt Side

- Files: `hw/rtl/top/soc_top_v2.sv`
- Prelearning: minimal
- Work: map all interrupt sources, MMIO regions, firmware-visible registers, and CPU-visible side effects; check them against `fw/main.c`
- Why this matters: software-visible architecture is part of RTL engineering, not a separate world

### Day 60: Finish The Full Draw.io Architecture Pack

- Files: no new mandatory RTL; use your notes and all major integration files
- Prelearning: none
- Work: produce final draw.io pages for: full system, address map, cache path, DRAM path, NoC mesh, tile internals, and firmware/software control flow
- Why this matters: if you cannot draw it cleanly, you probably do not understand it cleanly

### Day 61: Do One True Independent Rewrite

- Files: pick one subsystem to rebuild without looking: L1 cache path, DRAM front-end, router slice, or tile DMA gateway
- Prelearning: none
- Work: spend the full six hours building your own version from blank files and your notes only
- Why this matters: this is the point where learning becomes actual design ability

### Day 62: Compare Your Rewrite Against The Repo And Review Yourself Like A Senior Engineer

- Files: the subsystem you chose on day 61 plus the corresponding repo files
- Prelearning: none
- Work: diff behavior, interfaces, buffering strategy, edge cases, and timing-safe structure; write a blunt review of your own design
- Why this matters: self-review is how you stop being a student who only copies and start becoming an engineer who can judge design quality

### Day 63: Validate, Write, And Plan The Next Month

- Files: use the existing testbenches and your selected rewritten subsystem
- Prelearning: none
- Work: run the main lint/tests you know, write a three to five page architecture note, list the ten biggest things you learned, and choose the next subsystem you would now be comfortable designing mostly alone
- Why this matters: the goal is not just to finish a reading plan; the goal is to become capable of independent RTL design work

---

## Weekly Deliverables

At the end of each week, you should have the following:

- one updated draw.io page set
- one markdown note summarizing that week's architecture
- one independently rewritten module or module cluster
- one list of mistakes you made and then corrected after comparing to the repo

If you do not have those four outputs, you are reading too passively.

## What To Focus On When Comparing Your Rewrite To The Repo

Do not only ask "did I get the functionality right?" Ask these questions every time.

- Did I model the same interface contract?
- Did I accidentally create a combinational loop?
- Did I forget buffering, arbitration state, or backpressure handling?
- Did I ignore reset behavior?
- Did I miss alignment, width, or metadata details?
- Did I keep debug/perf visibility in the design?
- Did I choose a structure that would still scale when the system gets bigger?

## Coverage Summary

This roadmap covers the current larger mandatory RTL in these groups:

- `hw/rtl/top/`: days 2 to 9 and 57 to 59
- `hw/rtl/memory/`: days 10 to 12 and 30 to 31
- `hw/rtl/periph/`: days 13 to 14
- `hw/rtl/cache/`: days 15 to 23
- `hw/rtl/dram/`: days 24 to 29
- `hw/rtl/control/`: days 32 to 35
- `hw/rtl/noc/`: days 36 to 49 and day 55
- `hw/rtl/systolic/`: days 33 and 49 to 54
- `hw/rtl/monitor/`: day 55

## Final Thought

If you actually do this properly, six hours a day, seven days a week, and you really redraw, rewrite, compare, and document instead of skimming, then yes: over a few months this project can give you a very serious head start before university. It is not just a coding exercise. It is a compressed architecture, RTL, verification, and system-integration education.