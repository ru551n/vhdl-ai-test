# cnn_accel sizing proposal — 150 MHz, 60 FPS, 320x240

Status: **proposal, awaiting ratification.** No RTL or generic default has
been changed on the strength of this document.

Supersedes the FPS figures (not the structure) of
`cnn_accel_tiled_dataflow_proposal.md` §6, which were quoted at 100 MHz
for a 320x320 input.

## 1. Question asked

Can the accelerator be **scaled down** to speed up synthesis while still
delivering 60 FPS at 150 MHz on a 320x240 image?

**Answer: no.** The current `g_pe_rows = g_pe_cols = 8` array misses that
target by 1.72x. It is already, to within 14%, exactly a *30* FPS design
at these clock and frame dimensions. Scaling down is not available;
meeting 60 FPS requires scaling *up*. Separately, the synthesis-speed
motivation behind the question has already been solved by other means —
see §6.

## 2. Cycle budget

| Quantity | Value |
|---|---|
| Clock | 150 MHz |
| Frame rate target | 60 FPS |
| **Cycle budget per frame** | **2,500,000** |
| Frame time budget | 16.667 ms |

## 3. Workload at 320x240

320x240 has exactly 0.75x the pixels of the 320x320 case tabulated in
`cnn_accel_tiled_dataflow_proposal.md` §6 (same width, 240/320 the
height), and every layer's pixel count scales by that factor cleanly
(layer 1: 160x160 -> 160x120 = 19,200). Per-pixel PE cycles are
unchanged, since `ceil(9*in_c/g_pe_cols)` does not depend on frame size.

`OT = ceil(out_channels / g_pe_rows)` is the output-channel tile count,
i.e. the number of D6 ifmap passes.

| Layer | in_c | out_c | pixels | cyc/pixel | OT @8 | cycles @8x8 | OT @16 | cycles @16x8 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 16 | 19,200 | 4 | 2 | 153,600 | 1 | 76,800 |
| 2 | 16 | 32 | 4,800 | 18 | 4 | 345,600 | 2 | 172,800 |
| 3 | 32 | 32 | 4,800 | 36 | 4 | 691,200 | 2 | 345,600 |
| 4 | 32 | 64 | 1,200 | 36 | 8 | 345,600 | 4 | 172,800 |
| 5 | 64 | 64 | 1,200 | 72 | 8 | 691,200 | 4 | 345,600 |
| 6 | 64 | 128 | 300 | 72 | 16 | 345,600 | 8 | 172,800 |
| 7 | 128 | 128 | 300 | 144 | 16 | 691,200 | 8 | 345,600 |
| 8 | 128 | 256 | 75 | 144 | 32 | 345,600 | 16 | 172,800 |
| 9 | 256 | 256 | 75 | 288 | 32 | 691,200 | 16 | 345,600 |
| **Total** | | | | | | **4,300,800** | | **2,150,400** |

## 4. Verdict against the budget

| Config | MACs | Cycles/frame | Frame time @150 MHz | FPS | vs 60 FPS budget |
|---|---|---|---|---|---|
| 8x8 (today) | 64 | 4,300,800 | 28.67 ms | **34.9** | **1.72x over** |
| **16x8 (proposed)** | 128 | 2,150,400 | 14.34 ms | **69.8** | 86% of budget, **14% headroom** |
| 16x16 | 256 | 1,075,200 | 7.17 ms | 139.5 | 43% of budget |

8x8 does meet **30 FPS** at 150 MHz (4,300,800 vs a 5,000,000 budget, 14%
headroom). If 30 FPS is ever acceptable, today's array is already the
right size and nothing needs to change.

## 5. Why `g_pe_rows` is the axis to scale, not `g_pe_cols`

Both reach 128 MACs, but they are not equally invasive.

**`g_pe_rows` 8 -> 16 (recommended).** `g_pe_rows` is the
*output*-channel dimension, so raising it halves `OT` and leaves
per-pixel cycles alone. Every layer's `out_channels` is a multiple of 16
(layer 1 is exactly 16), so `OT` halves exactly with no rounding waste
and the total is exactly halved. Crucially it does **not touch the ifmap
path at all**: `g_tile_channels` stays 8, so `cnn_accel_window_gen`, its
64-bit word width, its 320-word row-tile invariant and its 3 BRAM are all
untouched, and the weight repacking order is unchanged. The change is
confined to `cnn_accel_pe_array` (twice the accumulator rows) and
`cnn_accel_weight_buffer` (twice the weight bits per pass).

**`g_pe_cols` 8 -> 16 (not recommended).** `g_pe_cols` is the
*input*-channel dimension and `g_tile_channels` is recommended to track
it. Raising it forces the window word to 128 bits, breaks the "exactly
320 cells per row for all nine layers" invariant that
`cnn_accel_tiled_dataflow_proposal.md` §1 deliberately exploits, changes
the weight gather order, and makes layer 1 (`in_c = 3`) waste half the
array. Strictly worse for the same MAC count.

**Bonus.** Halving `OT` also halves the D6 ifmap re-read traffic that the
2026-09-07 decision accepted: the ~7.15x re-read penalty drops to
~3.6x, roughly 3.9 MB/frame at 320x240 instead of 7.8 MB. This partially
repays the bandwidth cost of the always-re-stream decision.

## 6. The synthesis-speed motivation is already satisfied

The question paired "scale down" with "accelerate synthesis". Scaling down
is not needed for that, because the Vivado-only decision already
delivered it. Measured on this machine:

| Entity | Yosys | Vivado |
|---|---|---|
| `axi_stream_join` | 0.1 s | — |
| `cnn_accel_pool` | 2.0 s | — |
| `cnn_accel_weight_buffer` | 2.0 s | — |
| `cnn_accel_pe_array` | 3.5 s | — |
| `cnn_accel_window_gen` | 17 s | — |
| `cnn_accel_bias_requant` | 18 s | — |
| `cnn_accel_conv_core` | **~18 min** | **50 s** |

Every leaf is already under 20 s. `conv_core` was the only slow build and
Vivado does it ~20x faster. There is no synthesis-time problem left to
solve by shrinking the array.

If faster dev-loop iteration is still wanted later, the correct lever is a
**small dev generic set** (e.g. `g_pe_rows = g_pe_cols = 4`) registered as
an additional fast netlist build, keeping the authoritative 16x8 build for
real numbers. That decouples iteration speed from delivered performance
instead of trading one for the other. Not proposed for action here.

## 7. Resource impact of 16x8 — ESTIMATES, not measured

Flagged explicitly: these are scaled from the M7c measurements, **not**
Vivado results for a 16x8 build. They must be replaced with real numbers
before any checker is re-baselined.

| Resource | 8x8 measured (Vivado, xc7a200tfbg484-2) | 16x8 estimate | xc7a200t budget |
|---|---|---|---|
| DSP48E1 | 36 | ~64-72 | 740 (~9%) |
| RAMB36 / RAMB18 | 14 / 2 | ~28 / ~4 | 365 (~9%) |
| FFs (`conv_core`) | 2,824 | ~4,200-4,600 | 269,200 (~2%) |
| LUTs (`conv_core`) | 15,358 | ~26,000-30,000 | 134,600 (~20%) |

Nothing here is close to a limit on xc7a200t. The part was already raised
from xc7a100t for the 320x320 input; 16x8 does not threaten it.

The BRAM doubling is the one to watch: `cnn_accel_weight_buffer` must hold
weights for 16 output channels per pass instead of 8, so its 15 BRAM
roughly doubles. That is a direct undoing of half the M7b saving (72 -> 15
BRAM), which is worth stating plainly — but 15 -> ~30 out of 365 available
is not a concern at this part size, and the M7b rework is what made the
headroom available to spend.

## 8. Timing risk at 150 MHz — unassessed

All resource numbers to date come from **out-of-context netlist builds
with no timing closure** (`flow_status.md` M7c). 150 MHz has never been
constrained or met in this project; there is no timing report. Doubling
the accumulator rows widens the bias/requant fan-in, which is the most
likely critical path. **A constrained Vivado run is required before 150
MHz can be claimed at all**, at either array size. This proposal
establishes the cycle budget only, not timing feasibility.

## 9. Recommendation

1. Do **not** scale the array down.
2. Adopt `g_pe_rows = 16`, `g_pe_cols = 8` (128 MACs) as the target
   configuration for 60 FPS at 150 MHz on 320x240, with 14% cycle
   headroom.
3. Keep `g_tile_channels = 8`; the ifmap path and `window_gen` do not
   change.
4. Treat §7 as estimates; measure with Vivado before touching any
   resource checker.
5. Run a constrained 150 MHz build to retire the §8 timing risk before
   committing to the frame-rate claim.
6. Revisit only if 30 FPS becomes acceptable, in which case 8x8 already
   suffices and no change is needed.

## Open questions

- Is 60 FPS a hard requirement, or is 30 FPS acceptable? The answer
  decides whether any change happens at all.
- Is 150 MHz fixed, or is a higher clock available? 8x8 would need
  ~258 MHz to reach 60 FPS, which is unlikely to close on Artix-7 and is
  why the array, not the clock, is the lever.
- Does 320x240 replace 320x320 as the target input, or coexist with it?
  The layer table, `_WEIGHT_BUFFER_DEPTH = 288` and the
  `g_max_row_tile_words = 512` bound were all derived for 320x320 and
  remain valid for it; 320x240 only reduces pixel counts, so no bound
  needs to shrink.

## Implementation Notes (vhfill)
