# NoC Project Definition

## What Is Baseline

The baseline project is the regular 4x4 mesh NoC with the standard round-robin VC allocator and no in-network reduction enabled.

- Baseline transport: `noc_mesh_4x4` + `noc_router` + baseline `noc_vc_allocator`
- Baseline top-level configuration in `soc_top_v2`:
  - `SPARSE_VC_ALLOC = 0`
  - `INNET_REDUCE = 0`
- This is the current tapeout-safe configuration because it is the default in the live top and the end-to-end baseline regression still passes.

In plain terms: if someone asks "what is the chip by default?" the answer is "a normal mesh NoC accelerator SoC." The novelty features are optional compile-time experiments on top of that baseline, not the baseline itself.

## What Is Novelty

This project now has two novelty tracks and they should stay as the only two novelty claims.

### 1. Adaptive Sparse-Aware VC Allocation

This is the optional allocator in `noc_vc_allocator_sparse.sv`.

- It is now genuinely wired into the live router path.
- It tracks recent traffic mix and only reserves the highest VC when sparse-style traffic dominates.
- It treats `MSG_SPARSE_HINT`, `MSG_SCATTER`, `MSG_REDUCE`, and `MSG_BARRIER` as sparse-sensitive traffic.

What it is good for:

- protecting sparse/control traffic from dense bursts in some contention-heavy cases
- staying close to baseline behavior when traffic is not sparse-dominant

What it is not:

- it is not the baseline allocator
- it is not a big standalone performance story after RTL-model alignment

### 2. Router-Side In-Network Reduction

This is the optional `noc_innet_reduce.sv` path inside `noc_router.sv`.

- Intermediate routers can intercept real reduce flits and collapse multiple partial sums before they reach the root.
- The router-side path is now safer and more correct than before:
  - it only intercepts flits when it can actually accept them
  - it bypasses trivial one-contributor reductions instead of trapping them
  - it emits the accumulated flit on the routed output link instead of the tile-facing local output
  - it uses a dedicated reduction VC for emitted reduced traffic
  - it can derive a router-local subtree target from XY geometry and clamp it by the reduction group's expected contributor count
  - it now also accepts explicit per-group subtree target metadata through a sideband router configuration path keyed by `reduce_id`

What it is good for:

- reducing root-hotspot pressure
- cutting link traffic for sparse reduction-heavy phases
- making reduction traffic visibly smaller before it reaches the root

Current integration status:

- the live tile path now includes a root-side `tile_reduce_consumer` in `accel_tile.sv`
- `INNET_REDUCE = 0` gives a baseline end-to-end reduction path where all contributors still reach the root
- `INNET_REDUCE = 1` gives a router-reduced end-to-end path where routers emit subtree partials and the root consumer completes the reduction
- the explicit subtree metadata path is live in the router/mesh RTL and now exposed through the tile-array accelerator CSR window for software-visible programming
- `fw/hal_accel.h` now exposes helper functions to program and clear router-local reduction targets from firmware
- `soc_top_v2` still ties the external sideband metadata inputs off by default, but the accelerator CSR window can drive the live mesh internally when programmed
- `reduce_engine.sv` still exists as older reference logic, but it is not the block that is currently instantiated in the active tile path

Important limit:

- the validated end-to-end INR story now covers focused corner, edge, and interior XY roots, one explicit-metadata sparse-subset tree, a live full-SoC non-corner root case, and a live full-SoC CSR-programmed metadata case, but it is still not a blanket proof for arbitrary firmware-programmed trees or arbitrary sparse contributor subsets

## What Is Comparison-Only

These items should not be described as "the architecture" or "the chip."

- `noc_vc_allocator_static_prio.sv`
- `noc_vc_allocator_weighted_rr.sv`
- `noc_vc_allocator_qvn.sv`

Those are benchmark comparators only.

The same applies to older reduction/scatter blocks that exist in the repo but are not part of the live tile path today.

- `reduce_engine.sv` is a reference block, not the live instantiated root consumer in `accel_tile`
- `tile_reduce_consumer.sv` is the live instantiated root-side reduction sink in the current tile path
- `scatter_engine.sv` is also reference architecture, not part of the currently enabled top-level datapath

The benchmark scripts are also comparison infrastructure, not proof that a feature is fully integrated into the shipping SoC.

## What Is Actually Enabled In The Current Top

In the live source tree, the current top exposes both novelty knobs but leaves both off by default.

- `soc_top_v2.sv` exports `SPARSE_VC_ALLOC` and `INNET_REDUCE`
- both default to `0`
- those parameters propagate through `accel_tile_array` into `noc_mesh_4x4` and then into each router

So the current top is:

- novelty-capable
- baseline by default
- able to elaborate novelty-on builds for comparison work

Validation status at the time of this document:

- baseline `tb_soc_top` regression still passes `9 passed, 0 failed`
- focused end-to-end reduction RTL test now passes in both modes across representative root placements:
  - corner root `0`: baseline `root_packets=60`, `link_flits=192`, `cycles=80`; novelty `root_packets=8`, `link_flits=60`, `cycles=84`
  - edge root `1`: baseline `root_packets=60`, `link_flits=160`, `cycles=80`; novelty `root_packets=12`, `link_flits=60`, `cycles=72`
  - interior root `5`: baseline `root_packets=60`, `link_flits=128`, `cycles=80`; novelty `root_packets=16`, `link_flits=60`, `cycles=60`
- the corrected focused e2e sweep therefore shows about `73.3%` to `86.7%` fewer root-visible reduce packets and about `53.1%` to `68.8%` fewer inter-router flit-hops, while cycle impact ranges from `+5.0%` for the corner-root microbenchmark to `-25.0%` for the interior-root case
- focused explicit-metadata sparse-subset test now also passes:
  - subset `{1,4,5}` to root `0`, baseline `root_packets=12`, `link_flits=16`, `cycles=32`
  - subset `{1,4,5}` to root `0`, metadata-programmed INR `root_packets=8`, `link_flits=12`, `cycles=48`
- novelty-on full-SoC regression `tb_soc_top_inr` now passes with a non-corner root (`root=5`): `groups=4`, `root_packets=16`, `link_flits=60`, `expected_sum=115`
- full-SoC metadata regression `tb_soc_top_inr_metadata` now also passes through the accelerator CSR path:
  - subset `{1,4,5}` to root `0`, metadata CSR readback valid, `groups=4`, `root_packets=8`, `link_flits=12`, `expected_sum=10`
- paper-ready tables and charts for these corrected numbers now live in `docs/verification/INR_RESULTS.md`

## What To Tape Out Vs What To Publish

### What To Tape Out

Tape out the baseline NoC configuration.

- `SPARSE_VC_ALLOC = 0`
- `INNET_REDUCE = 0`

Reason:

- this is the cleanest end-to-end story
- it is the configuration already aligned with current handoff guidance
- the sparse allocator is low-risk but only modestly beneficial
- router-side INR is now end-to-end functional in the focused RTL path, but it is still the higher-risk research option compared with the default baseline top

### What To Publish

Publish the project as:

- a baseline mesh-NoC accelerator SoC
- with two optional research novelties:
  - adaptive sparse-aware VC allocation
  - router-side in-network reduction

The stronger publication story is the second novelty, not the first.

- Sparse-aware VC allocation should be presented as a secondary support mechanism.
- Router-side INR should be presented as the main architectural differentiator.

Be explicit in any paper, deck, or thesis that:

- the current integrated top defaults to baseline
- the router-side INR path is live, compile-time selectable, and validated in focused RTL, a novelty-on full-SoC regression, and a full-SoC CSR-programmed metadata regression
- the live root-side sink is `tile_reduce_consumer`, while `reduce_engine.sv` remains a reference block

## Where The Project Shines

The project is strongest in workloads with sparse, reduction-heavy communication where the problem is network pressure near a reduction root rather than raw dense streaming throughput.

Best-fit application patterns:

- K-split fully connected layers that need partial-sum reduction
- sparse all-reduce style phases across tiles
- reduction-heavy sparse MLP / classifier tails
- sparse statistics or aggregation phases where many one-flit contributions converge on a controller/root

Updated benchmark interpretation:

- Focused RTL end-to-end reduction sweep: `INNET_REDUCE=1` cuts root-visible reduce packets from `60` down to `8`, `12`, or `16` depending on whether the reduction root is a corner, edge, or interior node.
- The same sweep shows inter-router flit-hops collapsing from root-dependent baseline costs (`192`, `160`, `128`) to a fixed `60`, which matches the `15` tree edges needed to merge `16` nodes across `4` groups.
- Full-SoC INR regression with non-corner root (`root=5`): the live SoC consumes `16` root-visible partials for `4` groups, which matches the expected four inbound branches per group under XY reduction.
- Pure FC all-reduce: router-side INR removes about `73.3%` to `86.7%` of root-visible reduce packets and about `53.1%` to `68.8%` of total flit-hops, depending on root geometry.
- FC all-reduce with heavy DMA background: root-visible reduce packets still drop by about `86.7%`, but total flit-hop reduction falls to about `22.4%` and average modeled packet latency gets worse.
- Scatter-reduce style traffic: packet reduction remains strong, but overall link-traffic reduction is modest.

That means the project shines most when:

- sparse reduction traffic is a large part of the network load
- the main value is hotspot relief and traffic collapse
- dense DMA traffic is not completely dominating the fabric at the same time

## Autonomy Benchmark Direction

For external evaluation, the best benchmark direction is now six-camera BEV fusion modeled after `nuScenes mini`, not MNIST alone.

There are now two autonomy benchmark layers in the repo:

- a structured synthetic BEV-fusion traffic model inside `tools/noc_allocator_full_comparison.py`
- a trace-driven replay path using `data/noc_traces/nuscenes_mini_bev_frame.json` and `data/noc_traces/nuscenes_mini_bev_reduce.json`

Why this is the right benchmark:

- six camera branches create real multi-source fan-in instead of a toy single-stream CNN
- BEV fusion creates many-to-few aggregation that stresses the NoC near a fusion root
- the benchmark now looks closer to a perception/fusion SoC communication problem than an MNIST classifier problem

Current interpretation:

- the sparse-aware allocator remains close to neutral on this workload and should be treated as secondary
- router-side INR remains the strong result and cuts root-visible reduction traffic by about `66.7%` on the BEV reduction path

## Competition Demo Strategy

For a competition, use simulation as the proof vehicle and treat camera or video input as context, not as the primary metric source.

Recommended demo stack:

- a prerecorded driving clip or a simulator scene such as CARLA on the left side of the screen
- a simple mapping slide that shows six camera branches feeding BEV tiles on the 4x4 mesh
- a live baseline-vs-INR run of the NoC benchmark and, if time allows, a focused RTL counter demo showing root packets and flit-hops

What to avoid:

- do not make a live USB camera the core demo unless you already have a stable real-time ingest and feature extraction path
- do not lead with MNIST end-to-end numbers because they understate the network contribution
- do not oversell the allocator results because judges will see that they are modest

Best demo flow:

1. Show one autonomy frame or short simulated clip.
2. Show the packet trace / reduction groups derived from that frame.
3. Run baseline and INR side by side.
4. Show root-visible packets, flit-hops, and sparse-latency numbers.
5. Explain that the value is communication collapse near fusion roots.

If you want one sentence for the judges:

"We are not claiming a full self-driving computer; we are showing that an aggregation-aware NoC can materially reduce fusion-traffic pressure in a multi-camera BEV workload."

## Industry Relevance

This has real industry relevance if it is positioned correctly.

Relevant domains:

- multi-camera BEV fusion accelerators
- radar-camera or lidar-camera fusion fabrics
- zonal automotive SoCs with distributed perception blocks
- edge robotics systems with many sensors feeding a shared planner or fusion engine

What the impact is:

- lower hotspot pressure near fusion roots
- lower NoC traffic for reduction-heavy phases
- a path to lower interconnect energy per fused frame
- better scaling as the number of sensors or fused cells increases

What the impact is not:

- it is not proof that the full chip is already an automotive production SoC
- it is not a claim that every workload or every average-latency metric improves
- it is not a substitute for a full trace-driven perception pipeline or safety-qualified implementation

## Practical Positioning

If you need one sentence for the project:

"The baseline chip is a standard mesh-NoC sparse accelerator, and the publishable research contribution is an optional router-side sparse communication stack built from adaptive VC handling plus in-network reduction."

If you need one sentence for the tapeout decision:

"Tape out the baseline mesh, publish the two novelty options, and lead with router-side INR as the main research result."