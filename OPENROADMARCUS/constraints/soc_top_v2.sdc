set core_clk_port [lindex [get_ports clk] 0]
create_clock -name core_clk -period 20.0 $core_clk_port
set_clock_uncertainty 0.20 [get_clocks core_clk]

set core_input_ports [all_inputs]
set non_clock_inputs {}
foreach p $core_input_ports {
    if {$p != $core_clk_port} {
        lappend non_clock_inputs $p
    }
}

set_input_transition 0.10 $non_clock_inputs
set_input_delay 2.00 -clock [get_clocks core_clk] $non_clock_inputs

set_output_delay 2.00 -clock [get_clocks core_clk] [all_outputs]
set_load 0.05 [all_outputs]
set_false_path -from [get_ports rst_n]
