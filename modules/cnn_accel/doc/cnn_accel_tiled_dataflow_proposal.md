# cnn_accel tiled dataflow — design proposal (`vhdesign`)

Scope: `cnn_accel_window_gen`, `pe_array` (rewrite), `cnn_accel_weight_buffer`
addressing, and the `window_m2s_t`/`accum_m2s_t` records already added to
`cnn_accel_pkg.vhd`. Resolves the blocking flaw in `flow_status.md`/plan
D1: a beat cannot carry `K*K*C` taps because `axi_stream_pkg`'s `data` is
128 bits (16 int8 taps) while the target net needs 27..2304. No RTL here;
this is the contract for `vhfill`.

Numbers below use `g_pe_rows = g_pe_cols = 8`, `g_tile_channels = 8`
(rationale in §3/§6), 3x3 kernels, pad=1 all sides, 100 MHz, and the nine
target-network layers. Re-derive if these change.

## 1. Line buffer organisation

**Key fact exploited**: `in_width * in_channels` ("row bytes") is
invariant at 2560 B for every layer except layer 1 (960 B) — each
stride-2 halves width, doubles channels. With `g_tile_channels = 8`, cells
per row (`in_width * ceil(in_channels/8)`) come out to **exactly 320 for
all nine layers** (layer 1: `320*ceil(3/8)=320`; the rest: `in_channels`
is always a multiple of 8, so `in_width*in_channels/8` = 2560/8 = 320).

- **Word width**: `8 * g_tile_channels` bits = **64 bits** (one channel
  tile of one column). Not tied to the 128-bit AXI-Stream width — this
  is an internal memory, not a stream payload.
- **Depth**: new generic `g_max_row_tile_words`, bounding `in_width *
  ceil(in_channels/g_tile_channels)`. Minimum for this network: 320;
  recommend **384-512** for headroom / power-of-two addressing.
- **Banks**: unchanged, `g_max_kernel_size` (3), same ping-pong-by-
  `row mod g_max_kernel_size` scheme (`cnn_accel_window_gen.vhd:117-121`).
- **Capacity**: 384 x 64 b/bank ≈ 3072 B; 3 banks ≈ **9.2 KB** total
  (7.68 KB at the tight 320-word minimum — matches the brief's ~7.7 KB).
- **Addressing**: cell = `col*n_tiles + tile_idx`, `n_tiles =
  ceil(cfg_in_channels/Ct)` computed once at `start` (same one-shot style
  as `out_width_q`, `cnn_accel_window_gen.vhd:218-228`). Only the last
  tile of a layer may be partial (true for this network — §8 risk 3).
- **Write side (new)**: one accepted `s_stream` beat now writes one
  `(col, tile)` cell, not one full pixel — `n_tiles` beats/pixel on
  ingest instead of 1. `s_stream` stays plain `axi_stream_pkg` (128 b);
  `Ct` valid bytes sit low, rest `0`. `cnn_accel_axi_read_dma` (not yet
  built) must emit `n_tiles` beats/pixel — integration contract, §8 risk 7.
- **Read side**: `K_h` rows read in parallel (one BRAM port per bank);
  `K_w` columns per row are sequential through that one port. ~`K_w`
  (3) + 1-2 pipeline cycles per tile beat — far below `pe_array`'s own
  MAC cycles/beat (§3), so never the bottleneck.
- **BRAM inference (Artix-7)**: each row bank (384 x 64 b ≈ 24.6 kb) maps
  to one 36 Kb block configured 64-bit wide (standard Vivado inference,
  same hand-written-array/one-write-process/one-read-process idiom
  `cnn_accel_weight_buffer.vhd` already uses). **3 banks ≈ 3 BRAM36**
  total — trivial even on XC7A35T (50 blocks).

**Generic set** (replaces `g_line_buffer_channels`):

| Generic | Status | Meaning |
|---|---|---|
| `g_tile_channels` | replaces `g_line_buffer_channels` | Bound on runtime `Ct`. Recommend 8 (= `g_pe_cols`). |
| `g_max_row_tile_words` | new | Bound on `in_width*ceil(in_channels/g_tile_channels)`. 384-512 recommended. |
| `g_max_kernel_size` | unchanged | Row-bank count / tap-grid bound. |
| `g_max_fmap_width` | **retired** | Bounding the *product* directly avoids sizing width and channels independently (320 x 256 would need a 320-deep x 2048-bit row — ~80x the BRAM actually used). `cfg_in_width` is still range-checked against a bound derived from `g_max_row_tile_words`, not a dedicated generic. |

Retiring `g_max_fmap_width` for a product-bound generic is the single
biggest lever here: it is what keeps BRAM at ~3 blocks instead of ~80.

## 2. Beat sequencing

Output-pixel order unchanged (row-major, stride-aligned, today's
`out_row_q`/`out_col_q` walk). A new tile loop nests inside each pixel:

```
for out_row, out_col (existing row_ready-gated walk):
  for tile in 0 .. T-1:                      -- NEW
    emit one window_m2s beat:
      first_tile <= '1' when tile = 0
      last_tile  <= '1' when tile = T-1
      last       <= '1' when tile = T-1 AND (out_row,out_col) = final pixel
```

`first_tile`/`last_tile` are both `'1'` on the single beat when `T=1`.
`last` (frame framing) fires only on the final pixel's `last_tile` beat —
not on every `last_tile`. These flags are plain payload sidebands (Axi4.md
rule 25's shape — stable while `valid='1'`/`ready='0'`), not `TLAST`-style
framing.

**Padding/stride interaction: none, by construction.** The `row_ready`
spatial test (`cnn_accel_window_gen.vhd:328-340`) is channel-agnostic
already; tiling only changes how many beats follow once `row_ready` is
satisfied for a pixel, and every tile of a pixel reads the same
row/column range (zero-padding taps are `0` for every channel alike).
This orthogonality is why the retrofit (§9) never touches `row_ready`.

`window_valid`/`s_stream_s2m.ready` stay pure functions of registered
state (no combinational loop) but now gate on "last tile of the oldest
still-needed row consumed", not "the one pending window consumed".

## 3. `pe_array` microarchitecture

**Tap-to-MAC mapping**: one beat carries `K_h*K_w*Ct` taps (<=`9*8=72`
worst case). `g_pe_cols=8` columns process 8 taps/cycle; `g_pe_rows=8`
rows compute 8 output channels from the same 8 taps (broadcast-
activation / per-lane-weight — unchanged shape from the pre-tiling
proposal's §3.5/§6, only the tap *source* changes). Groups/beat =
`ceil(K_h*K_w*Ct/g_pe_cols)` = **exactly 9, no remainder**, at `K=3`,
`Ct=g_pe_cols=8` — the reason `g_tile_channels=g_pe_cols` is recommended.

**Cycles**: `groups_per_tile+1` per beat (same `run`-FSM shape as the
existing proposal's §5/§8, now re-entered `T` times per pixel). Per pixel:
`T*(groups_per_tile+1)`, dominated by `T*groups_per_tile` (§6 table).

**Partial-sum carry (the core rewrite)**:
- `first_tile='1'`: clear all `g_pe_rows` accumulators (replaces
  "clear on every window accept").
- Every beat's MAC sequence adds into the existing accumulators — no
  clear between tiles of one pixel.
- `last_tile='1'`: after the last group, accumulators hold the pixel's
  final int32 sums; commit to the one-entry output register and emit
  `m_accum_m2s` (`last` mirrors the window's `last`) — same commit logic
  as today, gated on `last_tile` instead of unconditionally.
- Accumulator register file: `g_pe_rows*g_accum_width` = **256 bits of
  flip-flops**, holding only the one in-flight pixel — never a whole-
  feature-map buffer. Layer 1's full-fmap partial-sum tensor would be
  `160*160*16*4 = 1,638,400 B ≈ 1.6 MB` (matches the brief) — infeasible
  on-chip; carrying the sum across only `T` beats/pixel in registers
  sidesteps this entirely.

**DSP48 inference intent**: each of the 64 lanes is one
`signed(7 downto 0)*signed(7 downto 0)` feeding an accumulate-add —
portable VHDL (`product := a*b; accum := accum + resize(product,32)`)
infers **one DSP48 per lane** (no manual dual-int8 packing — flagged as
a future item, §8 risk 4). **64 DSPs** for `g_pe_rows=g_pe_cols=8`: 71%
of XC7A35T's 90 DSP48E1 slices, 27% of XC7A100T's 240.

## 4. `weight_buffer` addressing

**No new port** is required — `weight_rd_addr` stays one flat row
address. What changes:

- `pe_array` resets `weight_rd_addr` to 0 only on `first_tile` of a new
  *pixel*, then runs continuously across all `T` tiles' groups before
  wrapping at the next pixel's `first_tile` (was: reset per window).
- `g_weight_buffer_depth` must grow from `groups_per_tile` to
  `ceil(K_h*K_w*in_channels/g_pe_cols)` — the whole layer's rows.
  Minimum across the network: **288** (layer 9); recommend **512**.

**Layout contract (the real new work, not RTL)**: the golden model's
weight layout is OHWI, `ic` fastest-varying *inside* `kr,kc`
(`cnn_accel_model.py:401,416`). `pe_array`'s row sweep is **tile-major**
(`tile_idx` hoisted above `kr,kc`). For `in_channels=32`, `Ct=8`, tile 1's
rows need `ic in 8..15` at *every* `(kr,kc)` — a strided gather across
the whole `kr,kc` range, not a contiguous slice. This repacking must
happen in the weight DMA fill path or a host-side compiler step — not yet
built (§8 risk 1). For this network, `groups_per_tile` is constant across
a layer's `T` tiles (only the last tile is ever partial), so no
variable-length-tile handling is needed here.

## 5. Output-channel tiling (D6)

`pe_array` computes 8 output channels/pass; `OT=ceil(out_channels/8)` is
2,4,4,8,8,16,16,32,32 for layers 1-9. v1 (D6): re-stream the whole ifmap
through `cnn_accel_window_gen` once per output-channel tile.

**Cost**: this does not waste PE cycles (each pass computes 8 genuinely
different channels — ~100% MAC utilization regardless). It costs DRAM
ifmap bandwidth: re-read `OT` times instead of once.

| | Ideal (1 read) | D6 (`OT` reads) |
|---|---|---|
| Total ifmap bytes, whole network | 1.46 MB | 10.44 MB |
| Ratio | 1x | **~7.15x** |

Compute time (§6, ~57 ms/frame) exceeds moving 10.44 MB at any realistic
DDR bandwidth (~21 ms at a conservative 500 MB/s, overlappable) — this
workload looks compute-bound, not DMA-bound, so D6's simplicity is likely
acceptable for v1. First optimization target if real bandwidth/power
proves tighter (§8 risk 6).

## 6. Cycle-count estimate

Per-pixel PE cycles = `ceil(9*in_channels/8)` (tiling-invariant here, no
rounding waste). Per layer = `pixels * cycles_per_pixel * OT`:

| Layer | in_c | out_c | pixels | cyc/pixel | OT | Layer cycles |
|---|---|---|---|---|---|---|
| 1 | 3 | 16 | 25,600 | 4 | 2 | 204,800 |
| 2 | 16 | 32 | 6,400 | 18 | 4 | 460,800 |
| 3 | 32 | 32 | 6,400 | 36 | 4 | 921,600 |
| 4 | 32 | 64 | 1,600 | 36 | 8 | 460,800 |
| 5 | 64 | 64 | 1,600 | 72 | 8 | 921,600 |
| 6 | 64 | 128 | 400 | 72 | 16 | 460,800 |
| 7 | 128 | 128 | 400 | 144 | 16 | 921,600 |
| 8 | 128 | 256 | 100 | 144 | 32 | 460,800 |
| 9 | 256 | 256 | 100 | 288 | 32 | 921,600 |
| **Total** | | | | | | **5,734,400** |

At 100 MHz: **57.3 ms/frame ≈ 17.4 FPS**. Sanity check: 5,734,400 * 64
MACs/cycle = 367.0M MAC-slots vs. 365.0M useful MACs computed directly
from network shapes — 0.5% `ceil()` overhead, i.e. ~99.5% PE utilization.
Scaling `g_pe_rows`/`g_pe_cols` trades DSPs for FPS roughly linearly.

**BRAM/DSP estimate (Artix-7, 8x8 array)**:

| Resource | Estimate |
|---|---|
| Line buffers | 3 x 384 x 64 b ≈ 9.2 KB ≈ 3 BRAM36 |
| Weight buffer, weight region | 2 x 512 x 512 b = 64 KB ≈ 15 BRAM36 |
| Weight buffer, bias region | 2 x 512 x 256 b = 32 KB ≈ 8 BRAM36 |
| **Total BRAM** | **~105 KB ≈ 24 BRAM36** (~48% of XC7A35T's 50; ~18% of XC7A100T's 135) |
| Partial-sum accumulators | 256 b, flip-flops only |
| DSP48 (MAC array) | **64** (~71% of XC7A35T's 90; ~27% of XC7A100T's 240) |

XC7A35T is DSP-tight already at this small an array; XC7A100T+ is safer
if `g_pe_rows`/`g_pe_cols` grow.

## 7. Backpressure/handshake

Per `shared/Axi4.md`'s mandatory-default rules (20/21/26), unchanged in
shape from today:

- **`s_stream`** (plain `axi_stream_pkg`): `ready` is a pure function of
  registered row-bank state, never of its own `valid` — same shape as
  `cnn_accel_window_gen.vhd:183`, re-scoped to the stricter "tile T-1 of
  oldest needed row consumed" condition (§2).
- **`window_m2s_t`/`s2m_t`** (new, D4): `valid` is pure registered state;
  `first_tile`/`last_tile` are stable sidebands, not framing bits. No
  loss/duplication (rule 21) applies per beat, as always.
- **`accum_m2s_t`/`s2m_t`**: unchanged from the pre-tiling proposal —
  `valid` is the one-entry output register, `last` mirrors the window's.
- **`weight_rd_addr`/`data`**: still a plain synchronous read port (no
  handshake), addressing change only (§4).

No new combinational-loop risk: every new signal (tile counter, `n_tiles`,
`T`) is registered, latched once per pixel or once per `start` — same
discipline as today's `out_width_q`/`kernel_h_q`.

## 8. Open questions / risks for the architect

1. **Weight repacking order** (§4) is a real strided gather, not a
   slice — needs an owner (host compiler vs. `cnn_accel_axi_read_dma`'s
   weight-fill path, neither built yet) before M9.
2. **Is `g_tile_channels = g_pe_cols` the right default?** Tunable, not
   forced by correctness — recommended here only for the clean 9-groups
   fit (§3).
3. **Partial non-final tiles unsupported.** Fine for this network (only
   the last tile of a layer is ever partial); a future network with
   irregular channel counts would need variable `groups_per_tile`/tile.
4. **DSP48 packing (2 int8 MACs/DSP)** deliberately not assumed (§3), to
   stay unambiguously portable. Worth a follow-up spike if 17 FPS is
   insufficient and DSPs (not fabric/BRAM) are the binding constraint.
5. **17.4 FPS at 8x8** — acceptable for the target application? Scaling
   trades DSPs for FPS; XC7A35T is already DSP-tight at 8x8.
6. **D6 restreaming's ~7.15x DRAM traffic** (§5) is argued compute-bound
   against an assumed, unmeasured 500 MB/s DDR figure — revisit once
   `cnn_accel_axi_read_dma` exists.
7. **`cnn_accel_axi_read_dma` (ifmap) must tile writes into
   `cnn_accel_window_gen`** (§1) — `n_tiles` beats/pixel instead of 1.
   That module's requirement doc predates this decision.
8. **Row-bank single read port -> ~K_w cycles/beat** (§1) assumed never
   the bottleneck; re-check if `g_pe_cols` grows large enough that
   `groups_per_tile` shrinks below `K_w`.
9. **`g_max_row_tile_words` is a hard elaboration bound** (sized
   384-512 against this network's 320-word need). A future layer with a
   larger product needs re-elaboration; recommend an elaboration-time
   assert (mirroring `g_max_kernel_size`'s) rather than silent corruption.

## 9. Rewrite or retrofit: `cnn_accel_window_gen`

**Retrofit, not rewrite.** The hardest, most bug-prone logic — the
`row_ready` fencepost test for arbitrary padding/stride
(`cnn_accel_window_gen.vhd:328-340`) — is completely unaffected by
channel tiling (§2's orthogonality argument) and is already green under
5 real tests covering exactly the expensive-to-re-derive corner cases
(asymmetric padding, non-dividing stride, mid-frame reset).

Required changes are localized and additive:

1. `row_bank_t` element width: `c_lane_width` from `8*g_line_buffer_
   channels` to a fixed `8*g_tile_channels` (§1).
2. Row bank depth/addressing: `g_max_fmap_width`-deep/one-cell-per-column
   -> `g_max_row_tile_words`-deep, `col*n_tiles+tile`-addressed (§1).
3. New `_q` registers at `start`: `tile_channels_q` (`Ct`), `n_tiles_q`
   (`T`), computed alongside the existing `out_width_q` arithmetic.
4. New innermost counter `tile_q` (`0..T-1`), advanced before
   `out_col_q`/`out_row_q` roll (§2).
5. Write-side (`control` process): one beat now writes one `(col,tile)`
   cell; row/column advancement accounts for `n_tiles` beats/pixel on
   ingest, mirroring item 4 on the write path.
6. `assemble_window`'s output type: `axi_stream_m2s_t`/`s2m_t` ->
   `window_m2s_t`/`s2m_t`, adding `first_tile`/`last_tile` (§2).

Estimated diff: ~120-180 lines against the current 374 — bounded and
mechanically scoped, not a ground-up redesign. The testbench's golden
model (`golden_window`/`golden_tap`) gains a channel-tile outer loop
rather than being replaced. A from-scratch rewrite would not simplify
anything: the padding/stride state machine is the module's actual
complexity, and tiling doesn't touch it — re-risking bugs already fixed
once (the FIFO-vs-full-row-bank lesson in this module's own proposal
doc §4) buys nothing. Treat the retrofit with rewrite-level review rigor
regardless: new `first_tile`/`last_tile`/write-side test cases are
mandatory before this module is green again.
