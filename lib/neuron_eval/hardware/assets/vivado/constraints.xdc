create_clock -period 10.000 -name clk [get_ports clk_100MHz]

set_property PROHIBIT TRUE [get_sites {AD6 W9 G4}]