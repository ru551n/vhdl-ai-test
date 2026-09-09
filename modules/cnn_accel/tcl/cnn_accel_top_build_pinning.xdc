# ------------------------------------------------------------------------------
# Constraints for the 'cnn_accel_top_build' / 'cnn_accel_top_build_pe_rows_16'
# top-level Vivado builds (see 'module_cnn_accel.py' and the header of
# 'src/cnn_accel_top_build.vhd').
#
# There is no target board. The only constraint that carries design meaning is
# the 150 MHz 'clk' -- the accelerator's specified operating frequency, and the
# thing this build exists to prove after place and route. Everything else here
# is the minimum Vivado needs to route and write a bitstream for the harness
# wrapper: five pins in one bank, with the harness' own I/O timing declared
# irrelevant.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# The design clock: 150 MHz, i.e. the target this whole build is measured
# against. Placed on a clock-capable (MRCC) pin of bank 14 so it can drive the
# global clock network without a placement error.
set_property -dict {"PACKAGE_PIN" "W19" "IOSTANDARD" "LVCMOS33"} [get_ports "clk"]
create_clock -name "clk" -period 6.667 [get_ports "clk"]

# ------------------------------------------------------------------------------
# The four harness pins. Bank 14, plain LVCMOS33; the choice of pin is
# arbitrary, they exist so that 'write_bitstream' has placed, I/O-standard-
# assigned ports to work with.
set_property -dict {"PACKAGE_PIN" "P20" "IOSTANDARD" "LVCMOS33"} [get_ports "reset"]
set_property -dict {"PACKAGE_PIN" "P22" "IOSTANDARD" "LVCMOS33"} [get_ports "stimulus"]
set_property -dict {"PACKAGE_PIN" "R22" "IOSTANDARD" "LVCMOS33"} [get_ports "result"]
set_property -dict {"PACKAGE_PIN" "P21" "IOSTANDARD" "LVCMOS33"} [get_ports "irq"]

# ------------------------------------------------------------------------------
# The four harness ports have no board-level timing contract to meet -- they
# belong to the synthetic harness, not to the accelerator -- but they must still
# be *constrained*, not waived: an input port with no clock relationship is a
# separate (unknown) clock domain to Vivado, and 'report_cdc' then raises a
# Critical "1-bit unknown CDC circuitry" violation that tsfpga fails the build
# on. That is the correct behaviour and is not something to switch off, so
# instead every harness port is declared system-synchronous to 'clk' with zero
# external delay. Vivado then times the pad-to-flip-flop and flip-flop-to-pad
# paths against the same 150 MHz clock as the rest of the design, and there is
# one clock domain in the whole build.
#
# Zero is the honest number here: there is no board, no external device and no
# trace, so there is no external delay to declare. It also makes these the
# *hardest* possible constraint on the harness paths, i.e. nothing is being
# relaxed to help the design pass.
set_input_delay -clock "clk" 0 [get_ports {"reset" "stimulus"}]
set_output_delay -clock "clk" 0 [get_ports {"result" "irq"}]
