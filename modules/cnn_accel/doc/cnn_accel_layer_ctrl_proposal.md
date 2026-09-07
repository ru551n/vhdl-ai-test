# cnn_accel_layer_ctrl — vhdesign proposal

Input: `modules/cnn_accel/doc/cnn_accel_layer_ctrl_req.md`.
Related: `doc/cnn_accel_arch.md` ("Reset policy", "Interface record policy",
ISA table), `doc/cnn_accel_tiled_dataflow_proposal.md` §5 (D6),
`doc/cnn_accel_weight_buffer.md`/`_proposal.md` §11-12 (single-buffer
rework), `doc/cnn_accel_axi_read_dma_req.md`, `doc/cnn_accel_ofmap_dma_req.md`.

**Ratified design decision this proposal is built on (user decision,
2026-09-07, not re-opened here):** the ifmap is ALWAYS re-streamed from
DDR on every output-channel-tile pass; there is no on-chip ifmap buffer.
A hybrid fits-in-BRAM ifmap buffer and an always-buffer scheme were both
explicitly considered and rejected, accepting up to `OT`x DDR read
traffic on late layers (D6) on the grounds that the accelerator is
expected to be DDR-read-bandwidth bound there. No `ifmap_buffer` module
exists or is proposed anywhere in this document.

## 1. Requirements summary

Executes exactly one decoded `layer_desc_t` end to end. Unlike the
requirement's current text (which runs `LOAD_WEIGHTS` once per *layer*),
this proposal runs the weight/bias fetch + ifmap re-stream + write-back
sequence once per *output-channel tile* — `OT = ceil(out_channels /
g_pe_rows)` times — per decision D6
(`doc/cnn_accel_tiled_dataflow_proposal.md` §5): `cnn_accel_pe_array`
computes only `g_pe_rows` output channels per pass, so a layer with more
output channels than that needs `OT` full passes, each with its own
weight/bias slice loaded into the (single-buffered, M7b) weight buffer and
its own full re-read of the ifmap. `POOL_MAX`/`POOL_AVG` are exempt from
tiling (§3.1) and always run exactly one pass.

Also folds in the two housekeeping fixes the requirement is stale on
independently of D6: the `weight_buffer_bank_sel` port and all ping-pong
language are removed (the buffer has been single-buffered since M7b,
`2d78d59`), replaced by the `fill_start`/`fill_is_bias` ports
`cnn_accel_weight_buffer` actually has.

## 2. Interface (as given by the requirement, with vhdesign additions marked)

### Generics

| Generic | Type | Purpose | Status |
|---|---|---|---|
| `g_max_kernel_size` | positive | validated against `layer_desc.kernel_h/w`/`pool_kernel_h/w` | requirement |
| `g_axi_addr_width` | positive | address width for `in_addr`/`out_addr`/`weight_addr`/`bias_addr` | requirement |
| `g_pe_rows` | positive | output channels computed per pass; sizes the `OT` tile loop and the per-tile bias byte length | **added** (§3.2) |
| `g_pe_cols` | positive | input-channel tile width assumed for the per-tile weight byte length | **added** (§3.2) |
| `g_accum_width` | positive := 32 | bias element width in bytes (`g_accum_width/8`) for per-tile bias addressing | **added** (§3.2), matches `doc/cnn_accel_arch.md`'s top-level generic of the same name |

### Ports

| Port | Dir | Type | Purpose | Status |
|---|---|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` | requirement |
| `layer_desc_m2s` | in | `cnn_accel_pkg.layer_desc_m2s_t` | from `cnn_accel_sequencer` | requirement |
| `layer_desc_s2m` | out | `cnn_accel_pkg.layer_desc_s2m_t` | | requirement |
| `layer_done` | out | `std_ulogic` | to `cnn_accel_sequencer` | requirement |
| `layer_error` | out | `std_ulogic` | to `cnn_accel_sequencer` | requirement |
| `weight_dma_req_m2s`/`s2m` | out/in | `cnn_accel_pkg.dma_req_*` | to weight `cnn_accel_axi_read_dma` | requirement |
| `ifmap_dma_req_m2s`/`s2m` | out/in | `cnn_accel_pkg.dma_req_*` | to ifmap `cnn_accel_axi_read_dma` | requirement |
| `ofmap_dma_req_m2s`/`s2m` | out/in | `cnn_accel_pkg.dma_req_*` | to `cnn_accel_ofmap_dma` | requirement |
| `wgen_cfg` | out | record (kernel_h/w, stride_h/w, pad_top/bottom/left/right, in_width, in_height, channels) | to `cnn_accel_window_gen` | requirement |
| `wgen_start`/`wgen_done` | out/in | `std_ulogic` | pulsed once per tile pass (§3.1), not once per layer | requirement, semantics extended |
| `route_sel` | out | `std_ulogic_vector(1 downto 0)` | opcode class to `handshake_splitter`/`handshake_mux` | requirement |
| `requant_cfg` | out | record (`bias_en`, `requant_en`, `relu_en`, `requant_scale`, `requant_shift`) | to `cnn_accel_bias_requant` | requirement |
| ~~`weight_buffer_bank_sel`~~ | ~~out~~ | ~~`std_ulogic`~~ | **removed** — dead ping-pong port, buffer is single-buffered since M7b | **removed** (§6, final section) |
| `wbuf_fill_start` | out | `std_ulogic` | pulses once per tile pass, directly to `cnn_accel_weight_buffer.fill_start` (replaces `weight_buffer_bank_sel`) | **added** (§3.2) |
| `wbuf_fill_is_bias` | out | `std_ulogic` | directly to `cnn_accel_weight_buffer.fill_is_bias`; `'0'` while the weight sub-request's stream is in flight, `'1'` while the bias sub-request's is | **added** (§3.2) |

`wbuf_fill_start`/`wbuf_fill_is_bias` are new direct point-to-point control
signals from `cnn_accel_layer_ctrl` to `cnn_accel_weight_buffer` — not
carried over the weight `cnn_accel_axi_read_dma`'s AXI4-Stream, which only
carries payload bytes. `doc/cnn_accel_weight_buffer_req.md`'s own Ports
table already documents `fill_start`/`fill_is_bias` as "from
`cnn_accel_layer_ctrl`"; the requirement for *this* module simply never
added the two output ports that drive them (a gap independent of the
`weight_buffer_bank_sel` staleness, though both point at the same M7b
rework). `doc/cnn_accel_arch.md`'s inter-module interface table
(`:286-307`) and block diagram (`:227-284`) do not show this edge either
— flagged here for `vharch` follow-up, not fixed unilaterally.

## 3. Design decisions (resolving requirement ambiguities / applying D6)

### 3.1 Which opcodes tile, and by how much

`OT = ceil(out_channels / g_pe_rows)` applies to `CONV2D`/`DWCONV2D`/`FC`
only: their output is produced through `cnn_accel_pe_array`, which
computes exactly `g_pe_rows` output channels per full ifmap pass
(`doc/cnn_accel_tiled_dataflow_proposal.md` §5). `POOL_MAX`/`POOL_AVG` do
not use `cnn_accel_pe_array` or `cnn_accel_weight_buffer` at all — pooling
preserves the channel count and `cnn_accel_pool` processes every
input-channel tile `cnn_accel_window_gen` emits within the single ifmap
pass already required to cover all channels (`cnn_accel_window_gen`'s own
internal `g_tile_channels` loop, orthogonal to and already covering the
full channel range in one `wgen_start`/`wgen_done` cycle) — so pooling's
`OT` is fixed at 1 regardless of `out_channels`. Concretely: `ot_last :=
is_pool(opcode) ? 0 : num_bits_needed`-sized `ceil(out_channels /
g_pe_rows) - 1`, computed once at `IDLE` from the latched descriptor.

### 3.2 New generics/ports for per-tile DMA addressing

The requirement's Generics/Ports tables give this module no way to
compute a per-tile byte offset or length, and no way to drive
`cnn_accel_weight_buffer`'s fill-session/region-select ports directly
(§2). Both gaps are new-to-this-module vhdesign additions, following the
same pattern `cnn_accel_weight_buffer_proposal.md` §3.1 used for
`g_accum_width`: `g_pe_rows`, `g_pe_cols`, `g_accum_width` are added as
generics (propagated from the same top-level generics of those names per
`doc/cnn_accel_arch.md`'s "Generic propagation map"), and `wbuf_fill_start`/
`wbuf_fill_is_bias` are added as ports (§2).

### 3.3 Per-tile weight/bias DMA addressing (algorithm, detailed in §7)

`cnn_accel_model.pack_weights_for_hw()`/`pack_bias_for_hw()` lay out a
layer's *whole* weight/bias image in DDR **output-channel-tile-major**:
`OT` consecutive blocks, one per tile, each block being exactly that
tile's `g_pe_rows`-wide weight rows (or bias row). This is precisely
shaped for D6's per-tile re-fetch: tile `ot`'s weight/bias sub-images are
each a single contiguous byte range within the packed image, so
`weight_addr`/`bias_addr` (the *first* tile's base addresses, exactly as
the ISA descriptor already defines them) plus `ot * <tile size>` is
sufficient addressing — no new descriptor field, no repacking at
`vhfill` time. §7 gives the exact byte-length formulas.

**Assumption carried over from `cnn_accel_conv_core.vhd:245-248` and
`doc/cnn_accel_weight_buffer.md`'s own row-layout contract**: the
per-tile weight byte length formula in §7 assumes the host compiler used
`tile_channels = g_pe_cols` when it called `pack_weights_for_hw()` (this
IP's established default, only elaboration-`assert`-checked as a
`report`-level mismatch warning at `conv_core`, not a hard requirement) —
if a future build uses a genuinely different `tile_channels`, this
module's addressing arithmetic and `cnn_accel_conv_core`'s own generic
would both need to agree on a real `g_tile_channels` generic, which
neither currently exposes to `cnn_accel_layer_ctrl`. Flagged, not
resolved here (same open-item style as `doc/cnn_accel_weight_buffer.md`'s
D10 lane-order risk).

### 3.4 Ofmap per-tile write-back — RESOLVED by decision S6 (channel-tiled ofmap planes)

**Historical note (the conflict this section used to flag).** Activations
were HWC (`out_h * out_w * out_channels` bytes, channel fastest-varying).
Under D6 one tile pass produces only `g_pe_rows` of those `out_channels`
channels, for *every* pixel, so writing just those bytes into their HWC
positions was **not** a contiguous `addr`+`length` range at any
granularity — the other `out_channels - g_pe_rows` bytes of every pixel
sit between one tile's bytes and the next. `dma_req_t` (`addr`, `length`
only) and the wrapped `dma_axi_write_simple` (a plain sequential burst
writer) have no stride/pitch field, and falling back to `out_h * out_w`
single-pixel requests (up to 25,600 for layer 1) would have defeated the
point of bursting. Three fixes were put to the architect: (a) add a real
strided-write capability to `dma_req_t`/`cnn_accel_ofmap_dma`, (b) make
the ofmap DDR layout tile-major and push a repack to the host, or (c)
accept per-pixel request granularity.

**Resolution: option (b), ratified as decision S6** (see
`doc/cnn_accel_arch.md`, "Off-chip activation layout", and
`modules/cnn_accel/flow_status.md`'s S1-S7 table). Activations in DDR are
**channel-tiled planes**:

```
ofmap[c_tile][y][x][t]        c_tile = 0 .. ceil(C/T) - 1,  t = 0 .. T-1
byte offset = ((c_tile * out_height + y) * out_width + x) * T + t
```

with **`T` fixed at 8, independent of `g_pe_rows`**. Consequences for this
module:

- **One plane is one contiguous range**, so `WRITE_OFMAP` issues ordinary
  `addr`+`length` `ofmap_dma_req`s. `dma_req_t` is **unchanged** and
  `cnn_accel_ofmap_dma` stays a thin `dma_axi_write_simple` wrapper — (a)
  is not needed, and (c)'s per-pixel granularity is avoided.
- **A pass emits `g_pe_rows / T` planes, hence that many requests.** At the
  shipped default (`g_pe_rows = 8 = T`) that is exactly one request per
  tile pass, as before. At `g_pe_rows = 16` (decision S4) it is **two
  ordinary requests**, not one double-length request — `T` must not track
  `g_pe_rows`, or the DDR layout and the host-side packing would depend on
  which bitstream is loaded (decision S1 makes `g_pe_rows` the only knob).
- **The ifmap is stored the same way**, so `STREAM_IFMAP` issues
  `ceil(in_channels / T)` requests per ifmap row rather than one whole-frame
  request (see §7). This is extra bookkeeping in this FSM — the price of
  removing the stride problem — and no new module.
- **The host repacks the network's final output once.** Intermediate
  layers never need repacking: every layer both writes and reads this
  layout.

`vhfill` implements the formulas in §7 directly; there is no longer an
open address-pattern item here.

### 3.5 Mid-layer host ABORT

Per `doc/cnn_accel_arch.md` "Reset policy" (`:60-96`), `ABORT` drives
`soft_reset_pulse` into the shared `reset_internal` tree; this module has
no ABORT-specific logic of its own beyond the ordinary synchronous
`reset` behavior already required for every stateful module in this IP
(§4). No `layer_done`/`layer_error` pulse is generated by a reset-induced
return to `IDLE` — a reset is not a completion, and the host that issued
`ABORT` already knows the layer did not finish.

## 4. Clock/reset

Single clock `clk`, synchronous active-high `reset` (`reset_internal`).
`reset` forces the FSM to `IDLE`, clears `ot_q`/`ot_last_q`/`drain_cnt_q`
to 0, deasserts every `*_dma_req_m2s.valid`, `wgen_start`, `wbuf_fill_start`,
and `layer_desc_s2m.ready`, and invalidates the latched descriptor
(irrelevant until the next accepted `layer_desc_m2s`) — this is exactly
what makes mid-layer host `ABORT` safe (§3.5): a partially-executed tile
loop can never be mistaken for a completed one after reset, matching
`cnn_accel_weight_buffer`'s own reset contract (its fill pointers reset
the same way, for the same reason). `wgen_cfg`/`requant_cfg`/`route_sel`
outputs are not reset (no completeness contract of their own; irrelevant
until the next `IDLE`→tile-loop entry re-latches them — reset
minimization gate).

## 5. Architecture and dataflow

```
layer_desc (sequencer) -> [latch @ IDLE] -> per-tile loop, OT times:
                                              |
                    +-------------------------+-------------------------+
                    |                                                    |
        LOAD_WEIGHTS (skipped for POOL_*)                                |
          wbuf_fill_start pulse -> cnn_accel_weight_buffer               |
          weight_dma_req(ot) -> weight cnn_accel_axi_read_dma -> WBUF    |
          bias_dma_req(ot)   -> weight cnn_accel_axi_read_dma -> WBUF    |
                    |                                                    |
        STREAM_IFMAP                                                    |
          ifmap_dma_req(addr=in_addr, SAME every tile, D6)               |
            -> ifmap cnn_accel_axi_read_dma -> cnn_accel_window_gen      |
          wgen_start ... wait wgen_done                                 |
          window_gen -> {pe_array | pool} -> bias_requant (or bypass)   |
            -> ofmap_dma  (this FSM does not touch per-pixel data)      |
                    |                                                    |
        WAIT_DRAIN (fixed pipeline-depth counter)                       |
                    |                                                    |
        WRITE_OFMAP                                                     |
          ofmap_dma_req(ot) -> cnn_accel_ofmap_dma                      |
                    |                                                    |
          ot_q < ot_last_q ? ---- yes ----> back to LOAD_WEIGHTS --------+
                    | no
                    v
                  DONE -> layer_done pulse -> IDLE
```

`cnn_accel_weight_buffer` is single-buffered (M7b): each tile's
`wbuf_fill_start` pulse overwrites the *same* region memories the
previous tile's pass just finished reading, which is why the fill
(`LOAD_WEIGHTS`) for tile `ot+1` must not start until `WRITE_OFMAP`
of tile `ot` has at least stopped reading through `weight_rd_addr`/
`bias_rd_addr` — guaranteed here because the FSM is strictly sequential
per tile (no tile's `LOAD_WEIGHTS` overlaps the previous tile's
`STREAM_IFMAP`/`WAIT_DRAIN`/`WRITE_OFMAP`). Overlapping tile `ot+1`'s
fetch with tile `ot`'s compute (to hide DMA latency behind the previous
tile's compute) is a real future optimization but is explicitly **not**
part of this v1 FSM — flagged as an open item (§10), not implemented,
to keep the state machine's correctness argument simple for `vhfill`.

## 6. State machine

States: `IDLE`, `LOAD_WEIGHTS`, `STREAM_IFMAP`, `WAIT_DRAIN`,
`WRITE_OFMAP`, `DONE` — the same six state names as the current
requirement, with the loop-back edge (§3, D6) added and
`weight_buffer_bank_sel` semantics replaced by `wbuf_fill_start`/
`wbuf_fill_is_bias`.

Registers driving the FSM: `layer_desc_q` (latched descriptor),
`ot_q`/`ot_last_q` (current/last output-channel-tile index, §7 for
widths), `drain_cnt_q` (fixed pipeline-depth countdown).

| State | Entry action / DMA issued | Transition | Condition |
|---|---|---|---|
| `IDLE` | On `layer_desc_m2s.valid='1'`: latch descriptor into `layer_desc_q`; assert `layer_desc_s2m.ready` for one cycle; compute `ot_last_q` (§3.1/§7); `ot_q <= 0`; configure `wgen_cfg`/`requant_cfg` from the latched fields; set `route_sel` from `opcode`. No DMA issued. | -> `LOAD_WEIGHTS` | `opcode` is `CONV2D`/`DWCONV2D`/`FC` |
| | (same latch/configure actions) | -> `STREAM_IFMAP` | `opcode` is `POOL_MAX`/`POOL_AVG` (tiling skipped, §3.1) |
| `LOAD_WEIGHTS` | On entry: pulse `wbuf_fill_start` for one cycle; issue `weight_dma_req_m2s` (`addr = weight_addr + ot_q*c_weight_tile_bytes`, `length = c_weight_tile_bytes`, §7) with `wbuf_fill_is_bias='0'`. When `weight_dma_s2m`'s `dma_done` fires: if `bias_en='1'`, issue `bias_dma_req` (reusing `weight_dma_req_m2s`/the same weight `cnn_accel_axi_read_dma` instance per `doc/cnn_accel_arch.md`'s submodule table — one instance serves both weight and bias sub-requests sequentially) with `addr = bias_addr + ot_q*c_bias_tile_bytes`, `length = c_bias_tile_bytes`, `wbuf_fill_is_bias='1'`. | -> `STREAM_IFMAP` | weight sub-request's `dma_done` seen, and (if `bias_en='1'`) bias sub-request's `dma_done` also seen |
| `STREAM_IFMAP` | Issue `n_ifmap_planes = ceil(in_channels/8)` successive `ifmap_dma_req_m2s` requests (`addr = in_addr + ct*c_ifmap_plane_len`, `length = c_ifmap_plane_len`, §7 — the S6 channel-tiled layout makes each input-channel tile its own contiguous plane), advancing a plane counter `ct_q` on each `dma_done`; pulse `wgen_start` on entry. The set of requests is **identical on every tile pass**, per D6: the ifmap is not sliced by output-channel tile. | -> `WAIT_DRAIN` | `wgen_done='1'` (all planes streamed) |
| `WAIT_DRAIN` | No DMA issued; `drain_cnt_q` counts down from a fixed pipeline-depth constant (`vhdesign`-time, covers `cnn_accel_pe_array`/`cnn_accel_pool`/`cnn_accel_bias_requant`'s combined pipeline latency) each cycle. | -> `WRITE_OFMAP` | `drain_cnt_q = 0` |
| `WRITE_OFMAP` | Issue `c_planes_per_pass = g_pe_rows/8` successive `ofmap_dma_req_m2s` requests (`addr = ofmap_addr + plane_idx*c_ofmap_plane_len`, `length = c_ofmap_plane_len`, §7) — one at the shipped `g_pe_rows = 8`, two at 16 (S6/S4). Each is an ordinary contiguous `addr`+`length` request; `dma_req_t` is unchanged. | -> `LOAD_WEIGHTS`, with `ot_q <= ot_q + 1` | last plane's `ofmap_dma_done`, and `ot_q < ot_last_q` |
| | (same DMA) | -> `DONE` | last plane's `ofmap_dma_done`, and `ot_q = ot_last_q` |
| `DONE` | Pulse `layer_done` for one cycle; `ot_q <= 0`. No DMA issued. | -> `IDLE` | unconditional (next cycle) |
| *any state* | Latch and pulse `layer_error` for one cycle; `ot_q <= 0`; deassert all `*_dma_req_m2s.valid`/`wgen_start`/`wbuf_fill_start`. | -> `IDLE` | any of `weight_dma_s2m.resp_error`, `ifmap_dma_s2m.resp_error`, `ofmap_dma_s2m.resp_error` pulses |

**Tile loop counter vs. the layer descriptor:** `ot_last_q` is computed
exactly once, at the `IDLE` -> `LOAD_WEIGHTS`/`STREAM_IFMAP` transition,
from `layer_desc_q.out_channels` (the ISA's W6 field, latched that same
cycle) and the `g_pe_rows` generic — `ot_last_q <= (opcode is POOL_*) ?
0 : ceil(out_channels_q / g_pe_rows) - 1` (§7 for the exact
unsigned-arithmetic form). `ot_q` then walks `0 .. ot_last_q` inclusive,
incrementing exactly once per `WRITE_OFMAP` -> `LOAD_WEIGHTS` back-edge,
so the loop runs exactly `OT = ot_last_q + 1` times regardless of opcode —
`OT = 1` for `POOL_*`/any layer with `out_channels <= g_pe_rows`, `OT =
ceil(out_channels / g_pe_rows)` otherwise. Neither `ot_q` nor `ot_last_q`
is ever compared against or derived from `in_channels` — input-channel
tiling (`T = ceil(in_channels/g_pe_cols)`) is a completely separate,
already-existing loop entirely internal to `cnn_accel_window_gen`/
`cnn_accel_pe_array` (their own `first_tile`/`last_tile` handshake,
`doc/cnn_accel_tiled_dataflow_proposal.md` §1-§2), invisible to and not
orchestrated by this FSM at all.

## 7. Algorithms

**Output-channel tile count** (§6):
```
ot_last_q <= is_pool(opcode_q) ? (others => '0')
                                : to_unsigned(ceil(out_channels_q / g_pe_rows) - 1, ...)
```
computed with `unsigned` division-by-power-of-two-or-general-divide per
§8 (widths), not a `real`/floating intermediate.

**Per-tile weight byte length** (§3.3, assuming `tile_channels = g_pe_cols`,
§3.3's flagged assumption), matching `pack_weights_for_hw()`'s per-tile
block size and `doc/cnn_accel_weight_buffer.md`'s own
`g_weight_buffer_depth` depth contract (`T*kernel_h*kernel_w` rows/tile):
```
n_in_tiles          = ceil(in_channels_q / g_pe_cols)
c_weight_tile_bytes = kernel_h_q * kernel_w_q * n_in_tiles * g_pe_cols * g_pe_rows   -- CONV2D/FC
```
`DWCONV2D`'s per-tile weight byte length is **not specified by any
existing document** (depthwise has no cross-input-channel accumulation, so
whether/how `n_in_tiles` applies to it is undefined in
`cnn_accel_pe_array`'s own requirement/proposal, and
`cnn_accel_model.py`'s `pack_weights_for_hw()` rejects the opcode rather
than invent a layout).

**Decision D2 (2026-09-07) removes this from the module's scope**: the
compiler rejects unsupported operations, so `DWCONV2D` never reaches the
accelerator. This module therefore computes **no** `DWCONV2D` weight
length, and — deliberately — adds no opcode-legality check, no
`layer_error` cause and no defensive decode for it either: the hardware is
not shaped around ops it does not implement. See `doc/cnn_accel_arch.md`,
"Opcode support status, and who rejects the unsupported ones". `vhfill`
implements the `CONV2D`/`FC` formula above and nothing else.

**Per-tile bias byte length** (§3.3, `pack_bias_for_hw()`'s per-tile block
is always exactly `g_pe_rows` int32 values):
```
c_bias_tile_bytes = g_pe_rows * (g_accum_width / 8)
```

**Per-tile weight/bias DMA addresses** (§6):
```
weight_dma_req.addr = layer_desc_q.weight_addr + ot_q * c_weight_tile_bytes
bias_dma_req.addr   = layer_desc_q.bias_addr   + ot_q * c_bias_tile_bytes
```

**Ifmap requests** (§3.4 / decision S6, identical every tile pass, §6).
The ifmap is stored as channel-tiled planes with `T = 8`, so the frame is
`ceil(in_channels / T)` separate contiguous planes rather than one
contiguous HWC frame. Per plane `ct = 0 .. n_ifmap_planes - 1`:
```
c_plane_channels  = 8                                   -- T, fixed (S6)
n_ifmap_planes    = ceil(in_channels_q / c_plane_channels)
c_ifmap_plane_len = in_width_q * in_height_q * c_plane_channels

ifmap_dma_req.addr   = layer_desc_q.ifmap_addr + ct * c_ifmap_plane_len
ifmap_dma_req.length = c_ifmap_plane_len
```
Total bytes streamed per tile pass are unchanged
(`in_width * in_height * in_channels`, modulo the zero-padding of a
partial final plane per D11); only the *number of requests* changes, from
1 to `n_ifmap_planes`. `STREAM_IFMAP` therefore carries a plane counter
alongside its existing per-pass logic and does not advance to
`WAIT_DRAIN` until the last plane's `dma_done`.

> Ordering note: `cnn_accel_window_gen` needs all `T` channels of a pixel
> in one beat, so planes are streamed **plane-major** (all of plane 0,
> then all of plane 1, ...) only when `in_channels <= T`; for
> `in_channels > T` the planes are consumed as the `T`-sized input-channel
> tiles the datapath already iterates over (`n_in_tiles` in the weight
> formula below), one plane per input-channel tile. That is the same loop
> the pre-S6 design ran; S6 only makes each tile's bytes contiguous.


**Ofmap output spatial size** (unchanged from the requirement's own
"standard conv output-size formula" note, needed for `WRITE_OFMAP`'s
per-tile `length`, §3.4):
```
out_width  = floor((in_width  + pad_left + pad_right  - kernel_w) / stride_w) + 1
out_height = floor((in_height + pad_top  + pad_bottom - kernel_h) / stride_h) + 1
```
**Ofmap write-back requests** (§3.4 / decision S6). Tile `ot` covers
output channels `ot*g_pe_rows .. ot*g_pe_rows + g_pe_rows - 1`, i.e.
`g_pe_rows / c_plane_channels` whole planes starting at plane index
`ot * g_pe_rows / c_plane_channels`. Per plane `p` of that pass:
```
c_ofmap_plane_len   = out_width * out_height * c_plane_channels
c_planes_per_pass   = g_pe_rows / c_plane_channels        -- 1 at 8 rows, 2 at 16
plane_idx           = ot_q * c_planes_per_pass + p

ofmap_dma_req.addr   = layer_desc_q.ofmap_addr + plane_idx * c_ofmap_plane_len
ofmap_dma_req.length = c_ofmap_plane_len
```
Both `c_plane_channels` (8) and `g_pe_rows` (8 or 16, decision S4) are
powers of two, so `c_planes_per_pass` is an elaboration-time constant and
`plane_idx` is a shift-and-add, not a divide. `WRITE_OFMAP` issues
`c_planes_per_pass` requests and only takes the `-> LOAD_WEIGHTS` back-edge
(or `-> DONE`) after the last one's `dma_done`.

This module does **not** special-case a final tile whose channel slice
overruns `out_channels` (`out_channels` not a multiple of `g_pe_rows`,
D11): it always writes full planes and relies on
`cnn_accel_bias_requant`'s already-existing zero-padded lanes (see
`doc/cnn_accel_weight_buffer.md`'s D11 note) to supply well-defined
(zero-weighted, computed but discardable) data for the padded lanes. The
host's one-off final repack (§3.4) drops them.

`POOL_MAX`/`POOL_AVG` are untiled (`OT = 1`) but still write the same
layout: `ceil(out_channels / c_plane_channels)` plane requests, since a
pool preserves channel count and therefore plane count.

## 8. Numeric types and widths

`ot_q`/`ot_last_q`: `unsigned`, width `num_bits_needed(c_max_ot - 1)`
where `c_max_ot` is a `vhdesign`-time constant bound (largest `OT` any
supported layer can produce — `ceil(65535/g_pe_rows)` in the worst case
from the ISA's 16-bit `out_channels` field, though the target network's
actual worst case is far smaller, 32 at layer 9). `drain_cnt_q`:
`unsigned`, width sized for the fixed pipeline-depth constant. All
per-tile byte-length/address arithmetic (§7) uses `unsigned` throughout
(`layer_desc_t`'s own fields are already `unsigned`/`signed` per
`cnn_accel_pkg.vhd:72-98`); intermediate products (`kernel_h_q *
kernel_w_q * n_in_tiles * g_pe_cols * g_pe_rows`) are sized generously
(the ISA's 32-bit address/length fields bound the final result, per
`shared/ModernVHDL.md`'s guidance to size for the contract, not the
common case) and never routed through `std_logic_vector` arithmetic
(`ieee.numeric_std` only, per the numeric type gate). `route_sel`,
`wbuf_fill_is_bias`, and every FSM state variable are plain
`std_ulogic`/an enumerated `state_t`, not encoded as raw integers.

## 9. Latency/throughput

One tile pass's latency is dominated by `STREAM_IFMAP` (a full ifmap
re-read + re-convolution, `in_width*in_height` cycles at
`cnn_accel_pe_array`'s own cycles/pixel rate,
`doc/cnn_accel_tiled_dataflow_proposal.md` §6) plus `WAIT_DRAIN`'s fixed
pipeline-depth constant plus each DMA's own latency
(`cnn_accel_axi_read_dma`/`cnn_accel_ofmap_dma`, out of this module's
control). A layer's total latency is `OT` times one tile pass's latency —
this is exactly D6's accepted cost (`~7.15x` aggregate ifmap DRAM traffic
across the target network, `doc/cnn_accel_tiled_dataflow_proposal.md`
§5), not a new regression introduced by this proposal. This FSM adds no
additional per-tile overhead beyond the DMA `dma_done` latencies and the
fixed `WAIT_DRAIN` count — `LOAD_WEIGHTS`/`STREAM_IFMAP`/`WRITE_OFMAP`
entries are single-cycle actions (issue-and-wait), matching the
un-tiled requirement's own per-state cost model.

## 10. Corner cases

- **`out_channels <= g_pe_rows`**: `ot_last_q = 0`, exactly one tile pass —
  degenerates to the pre-D6 requirement's behavior exactly (no change in
  observable timing for small layers).
- **`out_channels` not a multiple of `g_pe_rows`**: the final tile's
  padded lanes are handled entirely by `pack_bias_for_hw()`/
  `pack_weights_for_hw()`'s own D11 zero-padding (`doc/cnn_accel_weight_buffer.md`);
  this FSM issues the same fixed `g_pe_rows`-wide request every tile,
  including the last.
- **`POOL_MAX`/`POOL_AVG`**: `OT` fixed at 1 (§3.1); `LOAD_WEIGHTS` and
  the `wbuf_fill_start`/weight-DMA traffic are skipped entirely, exactly
  as the pre-D6 requirement already specifies for pooling opcodes — D6
  changes nothing about the pooling path.
- **`bias_en='0'`**: `LOAD_WEIGHTS` issues only the weight sub-request per
  tile, not the bias one — unchanged from the pre-D6 requirement, just
  now repeated `OT` times instead of once.
- **DWCONV2D's per-tile weight length**: **out of scope** per decision D2
  (2026-09-07) — the compiler rejects unsupported operations, so the
  opcode never reaches this module. No formula, no opcode-legality check
  and no `layer_error` cause is implemented for it (§3.3/§7).
  *Previously*: flagged as unresolved, since it needs
  `cnn_accel_pe_array`'s own depthwise-tiling contract, which does not
  exist in any requirement/proposal. That contract is still absent — D2
  makes its absence harmless rather than blocking.
- **Ofmap per-tile write-back address pattern**: **RESOLVED** by decision
  S6 (§3.4, formulas in §7). Was the single most consequential open item in
  this proposal; the channel-tiled `[C/8][H][W][8]` layout makes every
  write-back contiguous, so `dma_req_t` and `cnn_accel_ofmap_dma` are
  unchanged. New verification obligation instead of a blocker: cover both
  `c_planes_per_pass = 1` (`g_pe_rows = 8`) and `= 2` (`g_pe_rows = 16`),
  and an `in_channels` that is not a multiple of 8 (partial final ifmap
  plane, D11).
- **DMA `resp_error` from any of the three DMAs, at any FSM state**:
  handled uniformly (§6's "any state" row) — `layer_error` pulses and the
  FSM returns to `IDLE` regardless of which tile or which state it was in,
  matching the pre-D6 requirement's "any state" error handling exactly,
  just now also covering `LOAD_WEIGHTS`/`WRITE_OFMAP` re-entries on tiles
  2..`OT`.
- **Host `ABORT` mid-tile-loop**: covered by ordinary synchronous `reset`
  (§3.5/§4) — no special-case logic, no partial-tile state can survive a
  reset and be mistaken for progress.
- **`fill_start`/DMA overlap discipline** (§5): tile `ot+1`'s
  `wbuf_fill_start` never fires before tile `ot`'s `WRITE_OFMAP` has
  completed, by construction of the strictly sequential FSM — no explicit
  interlock signal is needed beyond the state ordering itself.

## 11. Selected patterns (`shared/DesignPatterns.md`)

- **FSM**: explicit enumerated `state_t`, one-hot vs. binary encoding left
  to `vhfill`/synthesis default; Moore-style outputs (`wgen_start`,
  `wbuf_fill_start`, DMA `req_m2s.valid`) driven from registered state,
  not combinationally from inputs, to avoid the long combinational chains
  the pattern doc warns against.
- **Counter**: `ot_q`/`ot_last_q`/`drain_cnt_q` are `unsigned` (not
  `natural`) since their bit width is part of this module's own interface
  contract with `g_pe_rows`/the ISA's 16-bit `out_channels` field, per the
  pattern doc's counter guidance ("use `unsigned` when bit-level
  width/overflow behavior is part of the implementation contract").
  Wrap/saturate behavior: `ot_q` never wraps — the FSM's own guard
  (`ot_q = ot_last_q`) prevents incrementing past the last tile.
- **Register bank / control bus**: `layer_desc_m2s_t`/`s2m_t`,
  `dma_req_m2s_t`/`s2m_t` are the project's existing typed-record pattern
  (`shared/DesignPatterns.md` "Packages and records" +
  `doc/cnn_accel_arch.md` "Interface record policy") — no new record
  types are introduced by this proposal beyond `wbuf_fill_start`/
  `wbuf_fill_is_bias`, which are plain `std_ulogic` (a single pulse and a
  single mode-select bit do not warrant a record).
- **Reset minimization**: per §4, only state that must survive an abort
  discovery cleanly is reset; `wgen_cfg`/`requant_cfg`/`route_sel` keep
  only declaration-initial values, consistent with this IP's own already-
  established reset-minimization convention (`cnn_accel_weight_buffer`'s
  read-data registers, `doc/cnn_accel_weight_buffer.md` "Clocking and
  reset").

## 12. AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

This module carries no AXI4 or AXI4-Stream signals directly — every
external interface is an internal control record (`layer_desc_*`,
`dma_req_*`) or a plain `std_ulogic`/small config record toward
`cnn_accel_window_gen`/`cnn_accel_bias_requant`/`cnn_accel_weight_buffer`.
The AXI4/AXI4-Stream rules in `shared/Axi4.md` are therefore the
responsibility of `cnn_accel_axi_read_dma` (x2, weight and ifmap) and
`cnn_accel_ofmap_dma`, which this module drives only through their
`dma_req_m2s_t`/`s2m_t` (`valid`/`ready`/`addr`/`length`) + `dma_done`/
`resp_error` control-plane, exactly per their own requirements
(`doc/cnn_accel_axi_read_dma_req.md`, `doc/cnn_accel_ofmap_dma_req.md`) —
this proposal introduces no new AXI4-adjacent signal and makes no new
AXI4 protocol decision. §3.4's ofmap addressing question, which used to be
the one AXI4-adjacent open concern here, is resolved by decision S6 without
touching `dma_req_t` or `cnn_accel_ofmap_dma`: it turned into "issue more
ordinary requests", not "issue a different kind of request". Multiple
successive requests per state change nothing about `shared/Axi4.md`'s
handshake-stability/burst-boundary/ordering rules — each request is an
independent, fully-handshaken `dma_req` and this module never has more than
one outstanding.

## 13. Verification plan

VUnit-5 testbench `tb_cnn_accel_layer_ctrl` (`architecture tb`). Since
this module has no AXI4/AXI4-Stream ports of its own, every interface is
driven/observed via small hand-written procedures over the internal
control records (`layer_desc_m2s`/`s2m`, `dma_req_m2s`/`s2m`,
`wgen_start`/`done`), matching this IP's existing testbench convention
for record-only modules. Test cases:

- `test_single_tile_layer` — `out_channels <= g_pe_rows`: exactly one
  `LOAD_WEIGHTS`/`STREAM_IFMAP`/`WAIT_DRAIN`/`WRITE_OFMAP` cycle, `OT=1`,
  bit-identical FSM timing to a hand-traced pre-D6 single-pass layer.
- `test_multi_tile_layer_ot_loop` — `out_channels` a clean multiple of
  `g_pe_rows` > 1 (e.g. `out_channels=32`, `g_pe_rows=8`, `OT=4`): verify
  `wbuf_fill_start` pulses once per tile, `weight_dma_req`/`bias_dma_req`
  addresses advance by `c_weight_tile_bytes`/`c_bias_tile_bytes` each
  tile, `ifmap_dma_req.addr` is identical every tile, and `layer_done`
  pulses only once, after the 4th tile's `WRITE_OFMAP`.
- `test_partial_last_tile` — `out_channels` not a multiple of `g_pe_rows`
  (e.g. 20 with `g_pe_rows=8`, `OT=3`): same per-tile request shape on
  the partial final tile as on full tiles (§10).
- `test_pool_opcode_skips_tiling` — `POOL_MAX`/`POOL_AVG`: `LOAD_WEIGHTS`
  never entered, `OT=1` regardless of `out_channels`.
- `test_bias_en_0_skips_bias_subrequest` — `bias_en='0'`: only the weight
  sub-request issues in `LOAD_WEIGHTS`, every tile.
- `test_resp_error_from_each_dma` (x3, weight/ifmap/ofmap) — `resp_error`
  from each DMA, at a tile other than the first, pulses `layer_error` and
  returns to `IDLE` without completing remaining tiles.
- `test_abort_mid_tile_loop` — `reset` pulsed mid-`STREAM_IFMAP` on tile
  2 of a multi-tile layer: FSM returns to `IDLE`, `ot_q=0`, no
  `layer_done`/`layer_error` pulse from the reset itself, and a
  subsequently issued fresh `layer_desc` runs its own full, correct tile
  loop from `ot=0`.
- `test_wait_drain_fixed_latency` — `WAIT_DRAIN`'s countdown is exactly
  the fixed pipeline-depth constant, independent of tile index.

No AXI4-Stream VC is needed (no such port exists on this module).
`cnn_accel_axi_read_dma`/`cnn_accel_ofmap_dma` are not instantiated in
this testbench — their `dma_done`/`resp_error`/`req_s2m.ready` are driven
directly by the test, per this IP's existing "isolated per-layer FSM
testable without a real instruction/DMA stream" design goal
(`doc/cnn_accel_arch.md` "Non-obvious boundary rationale", "kept
separate" bullet).

## Implementation Notes (vhfill)

(filled in during/after implementation)

## Proposed replacements for the hand-owned requirement text

`AGENTS.md` requires hand-owned requirement sections to be preserved, and
the user has said they will paste `cnn_accel_layer_ctrl_req.md`'s edits in
themselves — **`cnn_accel_layer_ctrl_req.md` is not edited by this
proposal.** The two blocks below are ready to paste as-is.

**Status: applied (decision D3).** The user authorized applying both
blocks; `cnn_accel_layer_ctrl_req.md` now carries them (the Ports table
gained both replacement rows, and the Functional Description additionally
carries a closing note that `DWCONV2D` is out of scope per decision D2).
The blocks are kept here as the record of what was pasted.

### (a) Ports table: delete the `weight_buffer_bank_sel` row

**Paste target:** `modules/cnn_accel/doc/cnn_accel_layer_ctrl_req.md`,
the Ports table (the row currently reads, verbatim, so it can be found
and deleted):

```markdown
| `weight_buffer_bank_sel` | out | `std_ulogic` | ping-pong bank select to `cnn_accel_weight_buffer` |
```

Delete this row. The buffer has been single-buffered since M7b
(`2d78d59`) — `cnn_accel_weight_buffer` has no bank-select port of any
kind any more (`doc/cnn_accel_weight_buffer.md`,
`src/cnn_accel_weight_buffer.vhd`). If the user also wants the two
replacement ports this proposal's own §2 adds (`wbuf_fill_start`,
`wbuf_fill_is_bias`) reflected in the requirement's Ports table (not
strictly required to unblock `vhfill`, since `vhdesign`-added ports are
already documented in the proposal), the two rows are:

```markdown
| `wbuf_fill_start` | out | `std_ulogic` | pulses once per output-channel-tile pass, to `cnn_accel_weight_buffer.fill_start` |
| `wbuf_fill_is_bias` | out | `std_ulogic` | to `cnn_accel_weight_buffer.fill_is_bias`; selects weight vs. bias sub-request routing |
```

### (b) Complete replacement `## Functional Description`

**Paste target:** `modules/cnn_accel/doc/cnn_accel_layer_ctrl_req.md`,
replacing its entire `## Functional Description` section (the text after
the `<!-- functional-spec: hand-owned below this line -->` marker) with
the block below.

```markdown
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
```
