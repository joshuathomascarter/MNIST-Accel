// filelist.f — Verilator source list for tb_soc_top
// Packages must come first so types are defined before use.
// SVA-only files (*_sva.sv, axi_protocol_sva.sv) are excluded;
// Verilator does not support full concurrent property SVA.

// ---- Packages ----
+incdir+/Users/joshcarter/MNIST-Accel/hw/rtl/top
+incdir+/Users/joshcarter/MNIST-Accel/hw/rtl/noc
+incdir+/Users/joshcarter/MNIST-Accel/hw/rtl/memory

/Users/joshcarter/MNIST-Accel/hw/rtl/top/soc_pkg.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_pkg.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/coherence_pkg.sv

// ---- Buffer ----
/Users/joshcarter/MNIST-Accel/hw/rtl/buffer/act_buffer.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/buffer/max_pool_2x2.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/buffer/output_accumulator.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/buffer/output_bram_buffer.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/buffer/output_bram_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/buffer/wgt_buffer.sv

// ---- Cache ----
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_data_array.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_tag_array.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_lru.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_cache_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_dcache_top.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l2_data_array.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l2_tag_array.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l2_mshr.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l2_cache_top.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/stride_prefetcher.sv

// ---- Control ----
/Users/joshcarter/MNIST-Accel/hw/rtl/control/barrier_sync.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/control/bsr_scheduler.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/control/clock_gate_cell.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/control/csr.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/control/tile_controller.sv

// ---- DMA ----
/Users/joshcarter/MNIST-Accel/hw/rtl/dma/act_dma.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dma/bsr_dma.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dma/dma_pack_112.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dma/out_dma.sv

// ---- DRAM ----
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_addr_decoder.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_bank_fsm.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_cmd_queue.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_refresh_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_scheduler_frfcfs.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_phy_simple_mem.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_write_buffer.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_deterministic_mode.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_power_model.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_ctrl_top.sv

// ---- HFT ----
/Users/joshcarter/MNIST-Accel/hw/rtl/hft/async_fifo.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/hft/eth_mac_rx.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/hft/eth_udp_parser.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/hft/fixedpoint_alu.sv

// ---- Host interface ----
/Users/joshcarter/MNIST-Accel/hw/rtl/host_iface/axi_dma_bridge.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/host_iface/axi_lite_slave.sv

// ---- MAC ----
/Users/joshcarter/MNIST-Accel/hw/rtl/mac/mac8.sv

// ---- Memory ----
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/boot_rom.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/sram_1rw_wrapper.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/sram_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/tlb.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/page_table_walker.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/snoop_filter.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/directory_controller.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/coherence_demo_top.sv

// ---- Monitor ----
/Users/joshcarter/MNIST-Accel/hw/rtl/monitor/perf.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/monitor/perf_axi.sv

// ---- NoC ----
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_route_compute.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_credit_counter.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_vc_allocator.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_vc_allocator_sparse.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_vc_allocator_qvn.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_vc_allocator_static_prio.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_vc_allocator_weighted_rr.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_switch_allocator.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_crossbar_5x5.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_input_port.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_innet_reduce.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_router.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_mesh_4x4.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_network_interface.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_bandwidth_steal.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_qos_shaper.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_traffic_gen.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/qos_arbiter.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/reduce_engine.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/scatter_engine.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/tile_reduce_consumer.sv

// ---- Peripherals ----
/Users/joshcarter/MNIST-Accel/hw/rtl/periph/uart_tx.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/periph/uart_rx.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/periph/uart_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/periph/timer_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/periph/gpio_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/periph/plic.sv

// ---- Systolic ----
/Users/joshcarter/MNIST-Accel/hw/rtl/systolic/pe.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/systolic/accel_scratchpad.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/systolic/systolic_array_sparse.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/systolic/tile_dma_gateway.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/systolic/accel_tile.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/systolic/accel_tile_array.sv

// ---- Top ----
/Users/joshcarter/MNIST-Accel/hw/rtl/top/axi_addr_decoder.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/axi_arbiter.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/axi_crossbar.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/obi_to_axi.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/soc_top_v2_asic_sim_wrapper.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/simple_cpu.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/soc_top.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/soc_top_v2.sv
