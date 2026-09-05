# cnn_accel_window_gen — proposal

## Requirements summary

Configurable `K_h x K_w`, stride `S_h x S_w`, zero-padded sliding-window
generator over row-major int8 input, per
`modules/cnn_accel/doc/cnn_accel_window_gen_req.md`: buffers `K_h - 1`
full rows plus the current row, and for every valid (stride-aligned)
output position emits one window of `K_h * K_w * g_line_buffer_channels`
int8 taps, zero-substituting any tap that falls in the padding region.
Generalizes the fixed-3x3, fixed-border-dilation pattern in
`modules/canny/src/canny_window3x3.vhd` to arbitrary, per-instruction
`K_h`/`K_w`/stride/padding — not a reusable instance of that entity (see
"Architecture and dataflow" below for why the generalization is not a
drop-in reuse of that module's technique). One instance, time-multiplexed
across instructions by `cnn_accel_layer_ctrl`; feeds both
`cnn_accel_pe_array` (`CONV2D`/`DWCONV2D`/`FC`) and `cnn_accel_pool`
(`POOL_MAX`/`POOL_AVG`) — the windowing operation itself is identical
between the two uses.

## Interface (copied from the requirement's structural section)

Generics: `g_max_kernel_size : positive` (upper bound on `K_h`/`K_w`),
`g_max_fmap_width : positive` (upper bound on row width), `g_line_buffer_channels
: positive` (channels processed in parallel per beat).

Ports: `clk`, `reset` (`std_ulogic`, `reset` = `reset_internal`);
`cfg_kernel_h`/`w`, `cfg_stride_h`/`w`, `cfg_pad_top`/`bottom`/`left`/`right`
(`std_ulogic_vector(7 downto 0)`, latched at `start`); `cfg_in_width`/
`height`/`channels` (`std_ulogic_vector(15 downto 0)`, latched at `start`);
`start` (in, pulse), `done` (out, pulse); `s_stream_m2s`/`s2m`
(`axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, raster-order input pixels);
`m_window_m2s`/`s2m` (`data` width `g_max_kernel_size^2 *
g_line_buffer_channels * 8`, only the low `K_h*K_w*channels*8` bits
meaningful).

## vhdesign addition: the fixed-width `axi_stream_pkg` data-field contract

Same fixed-128-bit `axi_stream_m2s_t.data` contract already recorded in
`doc/cnn_accel_pool_proposal.md`: this module's own `m_window_m2s.data`
must fit `g_max_kernel_size**2 * g_line_buffer_channels * 8 <=
axi_stream_data_sz` (128), checked by an elaboration-time `assert`. For
the pooling-path instance (`g_line_buffer_channels = 1`, per
`cnn_accel_pool_req.md`) this reduces to exactly `cnn_accel_pool`'s own
`g_max_kernel_size <= 4` contract — the two modules' generics must be
sized consistently by the top-level integrator, as already flagged in
`cnn_accel_pool_proposal.md`.

## Clock/reset behavior

Single clock, synchronous active-high `reset`, per `doc/cnn_accel_arch.md`
"Reset policy". `reset` clears `active_q` and the row/column position
counters (`cur_row_q`/`cur_col_q`/`out_row_q`/`out_col_q`) to a clean idle
state; the row-bank contents themselves are not reset (irrelevant while
`active_q = '0'`, and the next `start` begins overwriting them from
`(0, 0)` before any window can read stale content — no output is ever
produced without first writing at least the row(s)/column(s) it reads,
per the `row_ready` fencepost derivation below).

## Architecture and dataflow

```
s_stream (raster-order pixels)
  --> write row_banks(cur_row mod g_max_kernel_size)(cur_col)   (registered, on 'fire')
  --> advance cur_row_q/cur_col_q                                (registered, on 'fire')

row_banks (g_max_kernel_size full-row buffers, g_max_fmap_width wide each)
  --> assemble_window (comb.): for each of the K_h*K_w taps needed by
      (out_row_q, out_col_q), compute (input_row, input_col), test
      in-frame vs. [0,in_h)x[0,in_w), and either zero-fill (padding) or
      read row_banks(input_row mod g_max_kernel_size)(input_col)
  --> m_window_m2s.data / .valid (comb., = window_valid)

consume = window_valid and m_window_s2m.ready
  --> advance out_row_q/out_col_q, active_q, done   (registered, on 'consume')
```

### Design rationale: full-row banks, not FIFOs/shift registers

`canny_window3x3.vhd` builds its fixed 3x3 window from 2 elastic
line-buffer FIFOs (`hdl-modules` `fifo.fifo_wrapper`) plus 3-deep
column-tap shift registers per row-lane — a FIFO is popped exactly once
per value, in strict row order, and a shift register only ever holds the
last few columns. That is sufficient for canny because its window has no
configurable stride and no padding: every input sample is read by exactly
one window position, in strictly increasing raster order, so nothing is
ever needed twice.

This module's own padding and stride break that invariant in two
independent ways discovered during `vhtestgen`/`vhtestrun` (see
"Implementation Notes" below):

1. **Padding-induced replay.** With nonzero padding, an output position's
   real (unpadded) row/column range can be identical to a neighboring
   output position's range (e.g. a right/bottom-edge window whose
   padding clips its column range down to the same single real column
   an interior window before it also reads). A pop-once FIFO or a
   shallow column-tap shift register can no longer supply a value once
   its head has moved past it; a full-row *bank* — read (never popped)
   at whatever column address a tap needs — still can, however many
   output positions end up reading the same already-written cell.
2. **Stride not dividing the frame evenly.** When `stride` does not
   evenly divide `(in_h + pad_top + pad_bottom - kh)`, the last output
   row's real bottom row is *lower* than the input frame's last row,
   leaving one or more trailing input rows genuinely unread by the end
   of the frame — an ordering assumption a strict FIFO pipeline
   (designed around "every row is eventually read exactly once, in
   order") does not need to make, but that this module's variable
   stride/padding combination requires being robust to regardless.

The chosen design instead uses `g_max_kernel_size` full-row banks
(BRAM-inference intent, each `g_max_fmap_width` pixels wide). Physical
input row `r` always lives in bank `r mod g_max_kernel_size`; since
`cfg_kernel_h <= g_max_kernel_size` (asserted at `start`), at most
`g_max_kernel_size` distinct physical rows (the current row plus up to
`g_max_kernel_size - 1` previous ones) are ever simultaneously needed by
any pending/future output row, so this many banks never forces an unread
row to be evicted. Window assembly is a direct random-access read —
`row_banks(input_row mod g_max_kernel_size)(input_col)` — with no
pop/shift ordering constraint, so both failure modes above are handled
by construction rather than by special-casing them.

Elastic-stage classification (per `shared/DesignPatterns.md` "Ready/valid
elastic stage"): **not** a standard registered-ready single-entry stage —
`s_stream_s2m.ready` is instead gated by whether the currently-pending
window (if any) has been consumed, since a row bank must not be
overwritten while any tap of a not-yet-consumed window still needs to
read it (`s_stream_s2m.ready <= active_q and (not window_valid or
m_window_s2m.ready)`). `window_valid` depends only on registered state
(`row_ready`'s inputs are all `_q` signals plus the row banks
themselves), never on `s_stream_m2s.valid`/`m_window_s2m.ready`, so this
has no combinational loop and does not violate the same-channel
`valid`-depends-on-`ready` rule (see "AXI4/AXI4-Stream protocol
decisions" below).

## State machines

None — a single sticky `active_q` flag (frame in progress vs. idle) plus
four wrapping position counters (`cur_row_q`/`cur_col_q`/`out_row_q`/
`out_col_q`) implement the same effect with less state-explosion than an
enumerated FSM would, per `shared/DesignPatterns.md`'s counter-based-
sequencing preference.

## Algorithms

- **`row_ready` fencepost test** (window readiness): for output position
  `(orow, ocol)`, the row range needed is `[orow*sh - pt, orow*sh - pt +
  kh - 1]` and the column range is `[ocol*sw - pl, ocol*sw - pl + kw -
  1]` (may extend outside `[0, in_h)`/`[0, in_w)` — padding). Clip each
  range's upper bound to the real frame (`real_row_bot`/`real_col_right`);
  the window is ready once the *largest real row/column it needs* has
  been fully written: no real row at all (window is entirely vertical
  padding) → ready immediately; the needed row already fully written
  (`cur_row_q > real_row_bot`) → ready regardless of columns; still
  writing that exact row → also need its columns caught up (or no real
  column at all).
- **Window assembly**: for each of the `kh*kw` taps (row-major, `tap_idx
  = kr*kw + kc`), compute `(input_row, input_col)`, test in-frame against
  `[0, in_h) x [0, in_w)`, and either leave the tap's slice zeroed
  (padding) or read `row_banks(input_row mod g_max_kernel_size)
  (input_col)`.
- **Output dimension formula** (computed once per `start`, not a
  per-cycle datapath): `out_dim = (in_dim + pad_lo + pad_hi - kernel) /
  stride + 1` — plain integer division is deliberate here (see "Numeric
  types and widths").

## Numeric types and widths

- `cfg_kernel_h`/`w`, `cfg_stride_h`/`w`, `cfg_pad_*`: `std_ulogic_vector
  (7 downto 0)` at the port boundary, converted to `unsigned` internally
  for arithmetic (`kernel_h_q` etc.), per project convention.
- `cfg_in_width`/`height`/`channels`, `cur_row_q`/`col_q`, `out_row_q`/
  `col_q`, `in_width_q`/`height_q`, `out_width_q`/`height_q`: `unsigned
  (15 downto 0)` — wide enough for the largest frame/output dimension
  this IP's 16-bit `cfg_in_width`/`height` ports can express.
- `assemble_window`'s local row/column arithmetic (`row_top`, `col_left`,
  `input_row`, `input_col`, etc.): plain `integer` variables, not
  `unsigned` — these values are *signed* by construction (padding makes
  `orow*sh - pt` legitimately negative for top/left-edge windows), and
  the process only ever compares/tests them (in-frame test, `row_ready`),
  never feeds them into further registered arithmetic, so `integer` is
  idiomatic here (mirrors `canny_window3x3.vhd`'s identical choice for
  its own per-tap offset arithmetic).
- Row bank cell / window tap data: `std_ulogic_vector(c_lane_width - 1
  downto 0)` (opaque payload, `c_lane_width = 8 * g_line_buffer_channels`;
  no arithmetic performed on pixel values by this module).
- Output dimension division: plain `integer` division (`v_num_w /
  to_integer(unsigned(cfg_stride_w)) + 1`), computed once per `start`
  pulse (not a per-cycle datapath) — deliberate, not an accuracy
  concession, since `out_dim` is by definition an integer count of
  stride-aligned positions.

## Latency/throughput

Latency: variable, data-dependent — the first window becomes ready once
enough real rows/columns have been written to satisfy `row_ready`'s
fencepost test for `(0, 0)` (as little as zero input beats, for a window
whose entire real range is top/left padding; as much as
`kh - pt` full rows for a window with no padding at all). Not a fixed
per-instance constant like `canny_window3x3`'s `img_width + 1`, since
this module's variable padding directly shifts the threshold. Throughput:
the row-bank design allows accepting a new input beat only when either
no window is pending or the pending window has just been consumed
(`s_stream_s2m.ready <= active_q and (not window_valid or
m_window_s2m.ready)`) — full 1:1 throughput is achievable whenever
`m_window_s2m.ready` keeps up, since accepting an input beat and
consuming a window can happen the same cycle.

## Corner cases

- **1x1 kernel, any stride**: every output window is exactly one real
  tap (or zero-fill, for a padding-only position with a degenerate real
  range that never happens for a 1x1 kernel at valid padding since
  `kh - pt >= ...` — covered by `test_kernel_stride_shapes`'s `(1,1)`
  shape).
- **Non-square kernel/stride** (e.g. `2x3`, `3x2`): row and column
  fenceposts are independent, so no cross-term bug is structurally
  possible — covered directly by `test_kernel_stride_shapes`.
- **Padding large enough to clip a window's real range to a single row
  or column**: `has_real_row`/`has_real_col` and `real_row_bot`/
  `real_col_right` degenerate correctly to a single-element range —
  covered by `test_padding_all_sides`'s largest asymmetric case.
- **Stride not dividing the padded frame evenly**: the last output row's
  real bottom row can be lower than the input frame's last row, leaving
  trailing input rows genuinely unread (see "Design rationale" above) —
  the DUT correctly stops asserting `s_stream_s2m.ready` once its own
  `active_q` clears at the final window's acceptance, regardless of
  whether every raw input row was ever streamed.
- **Reset mid-frame**: `active_q` clears immediately, `s_stream_s2m.ready`
  and `m_window_m2s.valid` deassert the same cycle, and no `done` pulse
  fires for the aborted frame; a fresh `start` afterwards behaves
  identically to a first frame (row/column counters are unconditionally
  reset, not left in a partially-advanced state) — covered by
  `test_reset_mid_frame_abort`.

## Selected patterns (`shared/DesignPatterns.md`)

Counter-based sequencing (no FSM) for frame progress; direct random-access
row banks (plain arrays, BRAM-inference intent) rather than a reused
elastic FIFO primitive, per `shared/ReusableRTL.md`'s "write new RTL only
for behavior that doesn't already exist" — no existing reusable
primitive in this repo supports the replay-safe random-access read
pattern this module's padding/stride combination requires (see "Design
rationale" above).

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full backpressure on both `s_stream` and `m_window`. `m_window_m2s.valid`
(`window_valid`) is a pure combinational function of registered state
(`row_ready`'s inputs, `active_q`) — never of `s_stream_m2s.valid` or
`m_window_s2m.ready` — so it does not depend on its own channel's
`ready`, satisfying the same rule `canny_window3x3.vhd`'s "Interfaces/
protocols" documents. `s_stream_s2m.ready` does depend on
`m_window_s2m.ready` (via `window_valid`'s consumption), which is the
standard, allowed cross-channel "input readiness depends on output
readiness" elastic-pipeline shape, not the forbidden same-channel
violation.

## Verification plan

- `tb_cnn_accel_window_gen.vhd`, VUnit-5, hand-rolled record-port
  stimulus/monitor procedures (`axi_stream_pkg` records, not flat `t*`
  signals — same practical choice `cnn_accel_pool`'s/
  `cnn_accel_weight_buffer`'s own testbenches already made).
- Golden model: an independent full-frame array (`frame_t`) plus a
  from-scratch `golden_window`/`golden_tap` function pair, re-derived
  from the requirement doc directly (zero-fill for any `(row, col)`
  outside `[0, in_h) x [0, in_w)`), not calling into the RTL under test.
- Non-blocking scoreboard (`expected_q`, a `queue_t` of `(data, last)`
  pairs) enqueued up front by `enqueue_frame_expected` before a frame's
  pixels are even streamed — output order is a pure function of the
  configuration, independent of pacing — and popped/checked by an
  independent `monitor` process on every accepted `m_window` beat.
- Randomized independent stall on both `s_stream` (push side) and
  `m_window` (monitor's `ready` side), per `shared/Vunit.md`'s
  mandatory-backpressure default, plus a dedicated zero-stall
  `test_full_throughput` case.
- `test_kernel_stride_shapes`: sweeps `c_shapes` (`1x1` up to
  `c_kernel_max x c_kernel_max`, square and non-square) crossed with
  `c_strides` (`1x1`, `2x2`), no padding.
- `test_padding_all_sides`: sweeps `c_pads` (none, symmetric, two
  asymmetric combinations, one large enough relative to the 3x3 kernel
  to clip a window's real-row range to a single row), fixed `3x3`
  kernel/stride `1x1`.
- `test_backpressure`: two directed configs (`2x2`/stride `1x1` no
  padding; `3x3`/stride `2x2` with `1,1,1,1` padding) at 40% stall on
  both links.
- `test_full_throughput`: zero stall on both links.
- `test_reset_mid_frame_abort`: starts a frame, streams a fraction of
  its pixels, asserts `reset` mid-stream, confirms `s_stream_s2m.ready`/
  `m_window_m2s.valid`/`done` all read as idle immediately afterwards
  (no queued expectations for the aborted session, since it must never
  be allowed to complete/be checked), then runs one full ordinary frame
  afterwards to confirm no leftover state leaks in.
- Whole-simulation structural safety net (`done_relation_check`, a
  concurrent process, not one dedicated test): `done` must coincide
  exactly with the acceptance of the final (`last`) window
  (`m_window_m2s.valid = '1' and m_window_s2m.ready = '1' and
  m_window_m2s.last = '1'`), checked on every `done` pulse across every
  test in the same run.
- `c_kernel_max = 3` (not 2, the module's own minimum-supported generic):
  a max kernel size of 2 would leave some internal loop ranges
  degenerate (0/1 iterations), never exercising a genuine 3-tall/3-wide
  bank-index-wraparound case.

## Implementation Notes (vhfill)

Implemented directly (RTL and testbench authored together in this
session), then iterated once a real architectural bug was found during
`vhtestrun` — not a red-then-green fencepost-only fix, a genuine
generalization gap in the first design attempt:

- **Architectural rewrite (FIFOs/shift-registers → full-row banks)**:
  the first RTL pass, modeled directly on `canny_window3x3.vhd`, used
  `hdl-modules` FIFOs plus shallow (`g_max_kernel_size`-deep) column-tap
  shift registers. This compiled and passed simple no-padding,
  stride-1 cases, but deadlocked/mismatched once padding or non-dividing
  stride were exercised, for exactly the two reasons in "Design
  rationale" above (padding-induced replay of an already-fully-written
  row/column; stride leaving trailing input rows genuinely unread).
  Rewritten to the full-row-bank design described above; see that
  section for the complete rationale (recorded there rather than
  duplicated here, since it applies equally to both the proposal's
  intended design and the as-built RTL).
- **Testbench `stream_frame` deadlock**: the first testbench revision
  unconditionally streamed every raw input row, assuming the DUT always
  consumes the whole frame. When stride does not evenly divide `(in_h +
  pad_top + pad_bottom - kh)`, the DUT correctly deasserts
  `s_stream_s2m.ready` (via `active_q` clearing at the final window's
  acceptance) before every raw row arrives, so pushing the remaining,
  never-consumed rows blocked `push_pixel` forever. Fixed by computing
  `last_row_needed` (mirroring the RTL's own `row_ready` formula) in
  `run_frame` and streaming only up to that row.
- **`done_pulse_count` delta-cycle race**: `run_frame` read
  `done_pulse_count` (updated by a separate clocked process,
  `done_relation_check`) immediately after `drain_and_check`'s loop
  exited on the same rising edge `done` last pulsed, seeing the
  pre-increment value. Fixed with a `wait for 1 ns;` before the read to
  let delta cycles settle.
- **`done_relation_check` process structure**: reordered so `reset` is
  checked first (unconditionally clearing `done_pulse_count`), with the
  `done`-pulse-counting logic in an `else` branch — ensures the reset
  branch fires every reset cycle rather than being skipped by an
  earlier, differently-ordered condition.

No `--@` markers remain in either `src/cnn_accel_window_gen.vhd` or
`test/tb_cnn_accel_window_gen.vhd`. Final `vunit-mcp` result: all 5 tests
(`test_kernel_stride_shapes`, `test_padding_all_sides`,
`test_backpressure`, `test_full_throughput`,
`test_reset_mid_frame_abort`) pass under `vunit_compile`/`vunit_run_tests`
(`cnn_accel.tb_cnn_accel_window_gen.*`).
