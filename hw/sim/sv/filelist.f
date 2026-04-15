// filelist.f — Verilator source list for tb_soc_top (soc_top_v2)
// Packages must come first so types are defined before use.
// SVA-only files are excluded; Verilator does not support full concurrent property SVA.

// ---- Packages ----
+incdir+/Users/joshcarter/MNIST-Accel/hw/rtl/top
+incdir+/Users/joshcarter/MNIST-Accel/hw/rtl/noc
+incdir+/Users/joshcarter/MNIST-Accel/hw/rtl/memory

/Users/joshcarter/MNIST-Accel/hw/rtl/top/soc_pkg.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_pkg.sv

// ---- Cache ----
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_data_array.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_tag_array.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_lru.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_cache_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/cache/l1_dcache_top.sv

// ---- Control ----
/Users/joshcarter/MNIST-Accel/hw/rtl/control/barrier_sync.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/control/clock_gate_cell.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/control/tile_controller.sv

// ---- DRAM ----
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_addr_decoder.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_bank_fsm.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_cmd_queue.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_refresh_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_scheduler_frfcfs.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_phy_simple_mem.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_write_buffer.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/dram/dram_ctrl_top.sv

// ---- MAC ----
/Users/joshcarter/MNIST-Accel/hw/rtl/mac/mac8.sv

// ---- Memory ----
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/boot_rom.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/sram_1rw_wrapper.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/sram_ctrl.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/tlb.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/memory/page_table_walker.sv

// ---- Monitor ----
/Users/joshcarter/MNIST-Accel/hw/rtl/monitor/perf_axi.sv

// ---- NoC ----
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_route_compute.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_credit_counter.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_vc_allocator.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_vc_allocator_sparse.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_innet_reduce.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_switch_allocator.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_crossbar_5x5.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_input_port.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_router.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_mesh_4x4.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/noc/noc_network_interface.sv
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
/Users/joshcarter/MNIST-Accel/hw/rtl/top/simple_cpu.sv
/Users/joshcarter/MNIST-Accel/hw/rtl/top/soc_top_v2.sv
