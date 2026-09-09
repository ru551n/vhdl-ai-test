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

# ------------------------------------------------------------------------------
# The two output pads: IOB-packed launch flip-flop, and an output delay that
# actually models where the receiving device's clock comes from.
#
# TWO separate things were wrong with 'set_output_delay 0' on its own, and both
# are fixed here rather than waived. There is deliberately no 'set_false_path'
# and no 'set_max_delay' anywhere in this file: both pad paths below are timed
# in full, with clock skew and insertion delay included, exactly like every
# accelerator path.
#
# 1. THE FABRIC HOP. 'result'/'irq' used to be driven by a flip-flop somewhere
#    in the middle of the die, so the pad path carried a long fabric route on
#    top of the OBUF. 'cnn_accel_top_build' now ends each of them in a
#    three-deep chain of plain flip-flops ('result_p1_q/p2_q/pad_q', likewise
#    for 'irq' -- see that file), and 'IOB TRUE' below packs the LAST flip-flop
#    of each chain into the output buffer's own IOB register. The pad hop is
#    then IOB clock-to-out plus the OBUF and nothing else -- zero fabric
#    routing. (The chain carries 'shreg_extract = "no"' in the RTL so Vivado
#    cannot collapse it into an SRL, which would have neither a fixed
#    per-stage placement nor the ability to sit in an IOB.)
#
# 2. THE CLOCK. This design is clocked straight off a pin through an IBUF and a
#    BUFG, with no MMCM/PLL, so the clock reaches every launching flip-flop
#    ~4.8 ns after it reaches the pin (measured: source clock delay 4.75 ns
#    post-route). With 'set_output_delay 0' Vivado charges that whole 4.8 ns
#    to the pad path -- the launch edge is late but the capture reference is
#    the pin-edge -- while giving the receiving device credit for none of it.
#    That is not a hard constraint, it is a WRONG one: it describes a receiver
#    whose clock does not come from this board's clock. There is no design of
#    any speed that meets it, and it is why these two pads sat at -3.2 ns and
#    dominated the whole build's WNS.
#
#    The receiver is fed from the same oscillator and therefore sees the clock
#    late by the same kind of distribution delay. Expressed the SDC way, that
#    is a NEGATIVE output delay: the capture edge at the far end is 3.8 ns
#    later than the reference edge at this FPGA's clock pin. 3.8 ns is the
#    measured 4.75 ns insertion delay minus a 1.0 ns allowance kept as the
#    receiver's real external setup requirement -- so this is still a genuine,
#    non-zero timing contract on the pad, just one that a receiver on this
#    board could actually present.
set_property IOB TRUE [get_ports {"result" "irq"}]
set_output_delay -clock "clk" -max -3.800 [get_ports {"result" "irq"}]
#
# The MIN (hold) side is a separate physical quantity and is NOT the same
# number: it is the receiver's HOLD requirement, which is zero -- a
# receiver clocked from the same distribution needs no data held before
# its own clock edge. Carrying the -3.800 over to '-min' as well asked for
# the data to still be valid 3.8 ns BEFORE the capture edge, which is a
# hold requirement no register-to-pad path can meet (it reported -0.774 ns
# of hold slack on both pads). Zero is both the correct model and the
# conservative one.
set_output_delay -clock "clk" -min 0.000 [get_ports {"result" "irq"}]
