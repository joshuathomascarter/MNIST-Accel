# OPENROADMARCUS — Cleaned RTL for P&R

Top module: `soc_top_v2`  
Package: `rtl/top/soc_pkg.sv`, `rtl/noc/noc_pkg.sv`

## What's excluded (vs hw/rtl)
- `hft/` — unused HFT Ethernet block (not instantiated in soc_top_v2)
- `top/soc_top.sv` — old single-tile top (replaced by soc_top_v2)
- `top/accel_top.sv` — standalone accelerator-only top
- `top/zcu104_wrapper.sv` — FPGA-specific wrapper
- `top/soc_top_v2_asic_sim_wrapper.sv` — sim wrapper only
- `*_sva.sv`, `axi_protocol_sva.sv` — assertion-only files
- `memory/coherence_demo_top.sv`, `directory_controller.sv`, `snoop_filter.sv` — unused coherence IP
- Various unused NoC variants (`noc_vc_allocator_qvn`, `_sparse`, `_static_prio`, `_weighted_rr`)
- `noc_innet_reduce.sv`, `noc_bandwidth_steal.sv`, `noc_qos_shaper.sv`, `noc_traffic_gen.sv`
- `reduce_engine.sv`, `scatter_engine.sv`, `qos_arbiter.sv`

## Key files
| File | Description |
|------|-------------|
| `rtl/top/soc_top_v2.sv` | **Top module** |
| `rtl/top/soc_pkg.sv` | SoC-level parameters |
| `rtl/noc/noc_pkg.sv` | NoC parameters |
| `config.tcl` | OpenLane 2 config |
| `constraints/` | SDC timing constraints |
