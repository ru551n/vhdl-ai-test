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
| `wbuf_fill_start` | out | `std_ulogic` | pulses once per output-channel-tile pass, to `cnn_accel_weight_buffer.fill_start` |
| `wbuf_fill_is_bias` | out | `std_ulogic` | to `cnn_accel_weight_buffer.fill_is_bias`; selects weight vs. bias sub-request routing |

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
WAIT_DRAIN -> WRITE_OFMAP -> DONE`, with `WRITE_OFMAP` looping back to
`LOAD_WEIGHTS` once per output-channel tile (decision D6) before finally
advancing to `DONE`.

- `IDLE`: on `layer_desc_m2s.valid`, latch the descriptor, assert
  `layer_desc_s2m.ready` for one cycle, configure `wgen_cfg` and
  `requant_cfg` from the latched fields, set `route_sel` from `opcode`,
  and compute the tile count `OT = ceil(out_channels / g_pe_rows)` for
  `CONV2D`/`DWCONV2D`/`FC` (`OT = 1`, i.e. tiling skipped, for
  `POOL_MAX`/`POOL_AVG` — pooling preserves channel count and never uses
  `cnn_accel_pe_array`/`cnn_accel_weight_buffer`). Reset the tile counter
  to 0.
- The following states run once per output-channel tile, `OT` times per
  layer (D6, decision D6 in `doc/cnn_accel_tiled_dataflow_proposal.md`
  §5: `cnn_accel_pe_array` computes only `g_pe_rows` output channels per
  full ifmap pass, so a layer needing more output channels than that
  needs `OT` full passes, each re-streaming the whole ifmap from DDR —
  there is no on-chip ifmap buffer of any kind):
  - `LOAD_WEIGHTS` (`CONV2D`/`DWCONV2D`/`FC` only): pulse
    `cnn_accel_weight_buffer`'s `fill_start` (single-buffered since M7b —
    this pass's weight/bias set overwrites the buffer's one region pair
    in place, there is no bank to select), then issue `weight_dma_req`
    for the current tile's weight slice (`addr = weight_addr + tile_index
    * <this tile's weight byte length>`) while driving
    `cnn_accel_weight_buffer`'s `fill_is_bias='0'`; when `bias_en=1`, also
    issue a request for the current tile's bias slice (`addr = bias_addr
    + tile_index * g_pe_rows * 4` bytes) with `fill_is_bias='1'`; on both
    DMAs' `done`, go to `STREAM_IFMAP`.
  - `STREAM_IFMAP`: issue `ifmap_dma_req` (`addr=in_addr`, `length =
    in_width*in_height*in_channels` bytes — identical on every tile pass,
    the ifmap itself is never sliced by output-channel tile) and
    `wgen_start`; the ifmap stream flows `cnn_accel_axi_read_dma ->
    cnn_accel_window_gen -> {cnn_accel_pe_array | cnn_accel_pool} ->
    cnn_accel_bias_requant (or bypass, pool-max) -> cnn_accel_ofmap_dma`
    without this FSM touching per-pixel data; wait for `wgen_done`.
  - `WAIT_DRAIN`: wait for the downstream pipeline (PE array / pool /
    requant) to flush its last few pixels (fixed pipeline-depth counter,
    a `vhdesign`-time constant) before starting this tile's write-back.
  - `WRITE_OFMAP`: issue `ofmap_dma_req` (`addr=out_addr`, `length` sized
    for this tile's `g_pe_rows`-channel slice for
    `CONV2D`/`DWCONV2D`/`FC`, or the full `out_width*out_height*
    out_channels` bytes for `POOL_MAX`/`POOL_AVG` where `OT=1`); on its
    `done`, advance the tile counter — if this was not the last tile, go
    back to `LOAD_WEIGHTS` for the next tile; otherwise go to `DONE`.
- `DONE`: pulse `layer_done`, reset the tile counter to 0, return to
  `IDLE`. Any AXI error response surfaced by any of the three DMA
  requests, at any tile, at any state, pulses `layer_error` instead,
  resets the tile counter, and returns to `IDLE`.

Opcode support (decision D2): `DWCONV2D` is named above only because the
FSM treats it exactly like `CONV2D`/`FC` — an allocated opcode with no
ratified weight layout. It is **out of scope for the hardware**: the
compiler rejects it at `pack_weights_for_hw()`, so no `layer_desc` with
that opcode ever reaches this FSM, and this module deliberately performs
no opcode-legality check and raises no `layer_error` for it (see
`doc/cnn_accel_arch.md`, "Opcode support status").
