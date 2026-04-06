set core_clk_port [get_ports clk]
create_clock -name core_clk -period 20.0 $core_clk_port
set_clock_uncertainty 0.20 [get_clocks core_clk]

set non_clock_inputs [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]
if {[sizeof_collection $non_clock_inputs] > 0} {
	set_input_transition 0.10 $non_clock_inputs
	set_input_delay 2.00 -clock [get_clocks core_clk] $non_clock_inputs
}

set_output_delay 2.00 -clock [get_clocks core_clk] [all_outputs]
set_load 0.05 [all_outputs]
set_false_path -from [get_ports rst_n]
