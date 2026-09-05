# cnn_accel_layer_ctrl — requirement

## Responsibility

Executes exactly one decoded `layer_desc_t` end to end: programs
`cnn_accel_window_gen` (kernel/stride/pad, from W7/W8 for conv/dwconv/fc
or W11 for pool), triggers the weight/bias `cnn_accel_axi_read_dma`
instance when `opcode /= POOL_*`, triggers the ifmap
`cnn_accel_axi_read_dma` instance always, drives the opcode-selected
routing (`cnn_accel_pe_array` vs `cnn_accel_pool`, and
`cnn_accel_bias_requant`'s bypass/bias/scale/shift configuration from
`flags`/`requant_scale`/`requant_shift`), triggers `cnn_accel_ofmap_dma`
for the output run, and reports `layer_done`/`layer_error` once the
ofmap write-back completes.

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_max_kernel_size` | positive | validated against `layer_desc.kernel_h/w`/`pool_kernel_h/w` |
| `g_axi_addr_width` | positive | address width for `in_addr`/`out_addr`/`weight_addr`/`bias_addr` |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `layer_desc_m2s` | in | `cnn_accel_pkg.layer_desc_m2s_t` | from `cnn_accel_sequencer` |
| `layer_desc_s2m` | out | `cnn_accel_pkg.layer_desc_s2m_t` | |
| `layer_done` | out | `std_ulogic` | to `cnn_accel_sequencer` |
| `layer_error` | out | `std_ulogic` | to `cnn_accel_sequencer` |
| `weight_dma_req_m2s`/`s2m` | out/in | `cnn_accel_pkg.dma_req_*` | to weight `cnn_accel_axi_read_dma` |
| `ifmap_dma_req_m2s`/`s2m` | out/in | `cnn_accel_pkg.dma_req_*` | to ifmap `cnn_accel_axi_read_dma` |
| `ofmap_dma_req_m2s`/`s2m` | out/in | `cnn_accel_pkg.dma_req_*` | to `cnn_accel_ofmap_dma` |
| `wgen_cfg` | out | record (kernel_h/w, stride_h/w, pad_top/bottom/left/right, in_width, in_height, channels) | to `cnn_accel_window_gen` |
| `wgen_start`/`wgen_done` | out/in | `std_ulogic` | |
| `route_sel` | out | `std_ulogic_vector(1 downto 0)` | opcode class to the `handshake_splitter`/`handshake_mux` instances at `cnn_accel_top` (conv/dwconv/fc vs pool-max vs pool-avg) |
| `requant_cfg` | out | record (`bias_en`, `requant_en`, `relu_en`, `requant_scale`, `requant_shift`) | to `cnn_accel_bias_requant` |
| `weight_buffer_bank_sel` | out | `std_ulogic` | ping-pong bank select to `cnn_accel_weight_buffer` |

## Protocols

All internal control records/handshakes (`cnn_accel_pkg`); no direct AXI4
of its own.

## Clock/reset

Synchronous active-high `reset_internal`: the per-layer FSM state must be
abortable mid-layer by the host (per `doc/cnn_accel_arch.md` "Reset
policy").

<!-- functional-spec: hand-owned below this line -->

## Functional Description

FSM: `IDLE -> LOAD_WEIGHTS (skipped for POOL_*) -> STREAM_IFMAP ->
WAIT_DRAIN -> WRITE_OFMAP -> DONE`.

- `IDLE`: on `layer_desc_m2s.valid`, latch the descriptor, assert
  `layer_desc_s2m.ready` for one cycle, configure `wgen_cfg` and
  `requant_cfg` from the latched fields, set `route_sel` from `opcode`.
- `LOAD_WEIGHTS` (CONV2D/DWCONV2D/FC only): issue `weight_dma_req`
  (`addr=weight_addr`, `length` derived from `kernel_h*kernel_w*
  in_channels*out_channels` for CONV2D, `kernel_h*kernel_w*in_channels`
  for DWCONV2D, `in_channels*out_channels` for FC, all times 1 byte since
  weights are int8) into the *inactive* `cnn_accel_weight_buffer` bank;
  when `bias_en=1`, also issue a request for `bias_addr`
  (`out_channels` x int32); on both DMA `done`, flip
  `weight_buffer_bank_sel` and go to `STREAM_IFMAP`.
- `STREAM_IFMAP`: issue `ifmap_dma_req` (`addr=in_addr`, `length =
  in_width*in_height*in_channels` bytes) and `wgen_start`; the ifmap
  stream flows `cnn_accel_axi_read_dma -> cnn_accel_window_gen ->
  {cnn_accel_pe_array | cnn_accel_pool} -> cnn_accel_bias_requant (or
  bypass, pool-max) -> cnn_accel_ofmap_dma` without this FSM touching
  per-pixel data; wait for `wgen_done`.
- `WAIT_DRAIN`: wait for the downstream pipeline (PE array / pool /
  requant) to flush its last few pixels (fixed pipeline-depth counter,
  a `vhdesign`-time constant) before starting write-back, so
  `cnn_accel_ofmap_dma`'s `length` (`out_width*out_height*out_channels`
  bytes, derived from `in_*` and stride/kernel/pad per the standard conv
  output-size formula) is not issued before data exists.
- `WRITE_OFMAP`: issue `ofmap_dma_req` (`addr=out_addr`, computed
  `length`); on its `done`, go to `DONE`.
- `DONE`: pulse `layer_done`, return to `IDLE`. Any AXI error response
  surfaced by any of the three DMA requests at any state pulses
  `layer_error` instead and returns to `IDLE`.
