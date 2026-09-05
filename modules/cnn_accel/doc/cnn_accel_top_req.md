# cnn_accel_top — requirement

## Responsibility

Structural top only: instantiates `cnn_accel_csr`, `cnn_accel_sequencer`,
`cnn_accel_layer_ctrl`, three `cnn_accel_axi_read_dma` instances
(instruction / weight+bias / ifmap), `cnn_accel_ofmap_dma`,
`cnn_accel_weight_buffer`, one `cnn_accel_window_gen` instance,
`cnn_accel_pe_array`, `cnn_accel_pool`, `cnn_accel_bias_requant`, the
reused `axi.axi_simple_read_crossbar` (3:1 read arbitration), and reused
`common.handshake_mux`/`handshake_splitter` instances for opcode-selected
routing. No datapath logic of its own beyond generic/port propagation and
internal reset-tree combination (`reset_internal <= reset or
soft_reset_pulse`). See `doc/cnn_accel_arch.md` for the authoritative
block diagram, interface table and rationale.

## Generics

Same table as `doc/cnn_accel_arch.md` "Top-level generics" (authoritative
on conflict): `g_axi_addr_width`, `g_axi_data_width`, `g_axi_id_width`,
`g_axi_lite_addr_width`, `g_max_kernel_size`, `g_max_fmap_width`,
`g_line_buffer_channels`, `g_pe_rows`, `g_pe_cols`, `g_accum_width`,
`g_weight_buffer_depth`, `g_instr_word_bytes`.

## Ports

Same table as `doc/cnn_accel_arch.md` "Top-level ports": `clk`, `reset`,
`s_axi_lite_m2s`/`s2m` (`axi_lite_pkg` records), `m_axi_m2s`/`s2m`
(`axi_pkg` records), `irq`.

## Protocols

AXI4-Lite at `s_axi_lite`, AXI4 at `m_axi`. Internal links use
`hdl-modules` `axi_stream_m2s_t`/`s2m_t` records plus the new
`cnn_accel_pkg` records (`layer_desc_*`, `dma_req_*`) — see
`doc/cnn_accel_arch.md` "Inter-module interface table".

## Clock/reset

Single clock domain (`clk`). `reset_internal` (combining external `reset`
with `cnn_accel_csr`'s `soft_reset_pulse`) is fed to every submodule
except the register contents inside `cnn_accel_csr` itself, which use
unmodified `reset` — see `doc/cnn_accel_arch.md` "Reset policy".

<!-- functional-spec: hand-owned below this line -->

## Functional Description

Wires the submodules exactly per `doc/cnn_accel_arch.md`'s block diagram
and inter-module interface table. `cnn_accel_window_gen` is a single
instance, time-multiplexed: `cnn_accel_layer_ctrl` programs it per
instruction with either the conv/dwconv/fc kernel/stride/pad fields (W7/
W8) or the pool kernel/stride fields (W11), and its output stream is
routed by opcode (via `handshake_splitter`) to `cnn_accel_pe_array` or
`cnn_accel_pool`. `cnn_accel_bias_requant`'s input is opcode-muxed (via
`handshake_mux`) between `cnn_accel_pe_array`'s accumulator output and
`cnn_accel_pool`'s avg-sum output; `cnn_accel_bias_requant`'s output
converges with `cnn_accel_pool`'s max-path output (via `handshake_mux`)
before reaching `cnn_accel_ofmap_dma`.
