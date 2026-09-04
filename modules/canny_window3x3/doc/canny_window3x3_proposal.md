# canny_window3x3 — proposal

## Requirements summary

Generic reusable AXI4-Stream 3x3 sliding-window generator over a
continuous raster-scan stream, per
`modules/canny_window3x3/doc/canny_window3x3_req.md`. Full elastic
backpressure on both sides. Reused 4x in the canny pipeline with
different `g_data_width`. Emits, for every accepted input beat (once
primed), the 3x3 neighborhood centered on a position that lags the live
input by one row + a few columns, with correct zero-masking at frame
edges and a dilated (OR-reduced over all 9 taps) border flag.

## Interface (copied from the requirement's structural section)

Generics: `g_img_width : positive`, `g_img_height : positive`,
`g_data_width : positive`, `g_user_width : positive range 1 to 2`.

Ports: `clk`, `rst_n`, `s_axis_{tvalid,tready,tdata,tuser,tlast}`,
`m_axis_{tvalid,tready,tdata,tuser,tlast}` per the requirement's port
table. `s_axis_tuser` is `g_user_width` bits (bit0=SOF, bit1=border only
present when `g_user_width=2`); `m_axis_tuser` is always 2 bits
(bit0=SOF, bit1=border); `m_axis_tdata` is `9*g_data_width` bits, packed
row-major MSB-to-LSB (`w_tl` at the top, `w_br` at the bottom).

## Clock/reset behavior

Single clock, synchronous active-low reset. Reset clears all internal
line-buffer FIFOs (via their own `rst_n`-less design — see
"Implementation Notes" on how `fifo.fifo_wrapper` itself has no reset
port, addressed below), tap registers, and the output-side row/column
counters and priming flags.

## Architecture and dataflow

Reuses hdl-modules `fifo.fifo_wrapper` (synchronous mode,
`use_asynchronous_fifo => false`) as 2 elastic line-buffer FIFOs, per
`shared/ReusableRTL.md`, plus hand-written column-tap shift registers
and row/column bookkeeping (no existing reusable primitive matches the
windowing/masking/dilate behavior itself, per `shared/ReusableRTL.md`'s
"write new RTL only for behavior that doesn't already exist" rule).

```
                         (row R+1, live)                (row R)                 (row R-1)
s_axis ---+--> bottom-row 3-tap shift --+--> fifo1 (line buf, depth=W) --> mid-row 3-tap shift --+--> fifo2 (line buf, depth=W) --> top-row 3-tap shift
          |    (right=live, mid, left) |         (delay = g_img_width)   (right=live, mid, left) |         (delay = g_img_width)   (right=live, mid, left)
          |                            |                                                          |
        w_br,w_bm,w_bl                 (fifo1 write data = live bottom sample)                  w_mr,w_mm,w_ml                                          w_tr,w_tm,w_tl
```

- Each row-lane's "3-tap shift register" holds `{tdata, border-bit}` for
  the current (`right`, live/combinational), one-column-old (`mid`), and
  two-columns-old (`left`) samples of that row. The two FIFOs provide
  the row-to-row delay (1 line each); the shift registers provide the
  column-to-column delay within a row.
- `fifo1`/`fifo2` payload width is `g_data_width + 3`: `tdata`, a
  `border` bit (forced `'0'` internally when `g_user_width=1`, since no
  incoming border bit exists on `s_axis_tuser` in that configuration),
  plus `sof`/`tlast` passenger bits — carried through *only* to be picked
  off at the exact center-tap position (`mid_mid`, i.e. the mid-row's
  `mid` tap) for the output's `tuser(0)` passthrough; not used by any of
  the other 8 taps. This satisfies the requirement's "ride alongside
  ... through the same elastic lanes" rule for SOF without a second,
  independently-clocked delay path that could drift from the data path.
- Depth of each `fifo.fifo_wrapper` instance is
  `round_up_to_power_of_two(g_img_width + 1)` (hdl-modules `math_pkg`),
  not `g_img_width` itself — see "Corner cases" below on why the `+1`
  headroom is required (avoids `fifo.vhd`'s documented one-cycle
  write-side pessimism when full+read-same-cycle) and why `fifo_wrapper`
  needs a *power-of-two* depth regardless of `g_img_width`'s own value.
  A dedicated internal counter (not the FIFOs' own fullness/level output)
  decides exactly when to start reading each line buffer, so the extra
  FIFO headroom is pure margin, never load-bearing for correctness.
- A single shared `fire` condition (`s_axis_tvalid and s_axis_tready`)
  gates every internal register/FIFO write+read for that cycle — the
  whole pipeline advances in lockstep by construction (this is what
  makes "no bubble insertion, no re-fetch, no dropped/duplicated sample"
  true under backpressure): once primed, `fire` is simultaneously "one
  accepted input beat" and "one accepted output beat."
- `s_axis_tready <= down_ready and fifo1.write_ready and (fifo2.write_ready or not mid_row_active)`,
  where `down_ready <= (not (primed or draining)) or m_axis_tready` —
  before the window is primed (or draining), downstream readiness is
  irrelevant (nothing is being offered yet); once primed or draining,
  backpressure propagates upstream in the usual elastic-pipeline way.
  **See "Amendment" below** — the post-proposal implementation renamed
  the third sticky flag from `window_valid` to `primed` and added a
  fourth, non-sticky `draining` state; the actual `m_axis_tvalid`
  (`window_valid`) is *not* one of the sticky flags described in the next
  bullet.
- Priming is tracked by a single saturating `accepted_beats` counter and
  3 sticky (once-set, never-cleared) flags derived from it:
  `mid_row_active` (fifo1 starts being read), `top_row_active` (fifo2
  starts being read), `primed` (see Amendment below — `primed` alone is
  *not* `m_axis_tvalid`). Per the requirement, output becomes available
  once `2*g_img_width + 2` accepted input beats have occurred — **the
  implementation deviates from this**; see "Implementation Notes
  (vhfill)" and "Amendment" below.
- Output-side `out_row`/`out_col` counters (0-based) are pure functions
  of the `fire` history (advance only on an actual accepted beat, so
  they cannot desync under backpressure — see "Numeric types and
  widths"), reset by `rst_n`, not by `s_axis_tuser(0)`/`tlast` directly;
  a simulation-only assertion cross-checks that they read back `(0, 0)`
  whenever the center tap's own passed-through SOF bit fires, as an
  independent consistency check on the "recomputed from the stream of
  accepted `s_axis_tlast`/SOF pulses" requirement language.
- `m_axis_tlast <= to_sl(out_col = g_img_width - 1)` — recomputed, not
  passed through (per requirement).
- `m_axis_tuser(0) <= mid_mid.sof` — passed through, delay-matched to the
  center tap via the same elastic path as the data (per requirement).
- `m_axis_tuser(1) <= edge_here or (OR of the 9 taps' border bits, each
  taken as `'0'` for any tap whose row/col offset places it outside the
  frame relative to `(out_row, out_col)`)` — the "growing border" dilate,
  see `doc/canny_arch.md` "Growing border (corrected, rev 2.2)".
- `m_axis_tdata`'s 9 slices: each tap's `g_data_width` bits are the tap's
  real stored `tdata` when in-frame, else `(others => '0')` (per
  requirement, "out-of-bounds window taps ... read as `(others => '0')`").

## State machines

No explicit FSM; a small set of sticky priming flags
(`mid_row_active`/`top_row_active`/`window_valid`) plus two wrapping
counters (`out_row`/`out_col`) implement the same effect with less
state-explosion than an enumerated FSM would, per
`shared/DesignPatterns.md`'s counter-based-sequencing preference over an
FSM when the sequencing is a simple linear/wrapping progression.

## Algorithms

3x3 neighborhood extraction via 2 chained line-buffer FIFOs + column
shift registers (see dataflow diagram above); per-tap in-frame test via
row/column offset comparison against `(out_row, out_col)`; border dilate
via 9-way OR-reduction of (masked) incoming border bits OR'd with this
module's own edge-of-frame test.

## Numeric types and widths

- `tdata`/window taps: `std_logic_vector` (opaque payload, no arithmetic
  performed on pixel values by this module).
- `out_row`/`out_col`: plain `natural range 0 to g_img_height-1` /
  `0 to g_img_width-1` — small enough range that `natural` (rather than
  `unsigned`) is idiomatic for a simple wrapping position counter used
  only in comparisons, per project convention of using `unsigned` for
  arithmetic *results* that feed into further arithmetic; these counters
  are compared, not added/subtracted with pixel data.
- `accepted_beats`: `natural range 0 to 2*g_img_width+1` (saturating) —
  large enough to reach the priming threshold and no further; kept small
  deliberately rather than a wide free-running counter.
- FIFO payload width `g_data_width + 3`: `std_logic_vector`, packed via
  local pack/unpack functions (`tlast & sof & border & tdata`, MSB to
  LSB) — chosen over a record type since `fifo.fifo_wrapper`'s own
  `write_data`/`read_data` ports are `std_logic_vector`.

## Latency/throughput

Latency: `2*g_img_width + 2` accepted input beats before the first valid
output beat (per requirement); one-time per instance (subsequent frames,
if the stream continues without draining, do not re-incur this fill
delay since the line buffers keep flowing continuously). Throughput: 1:1
with `s_axis` once primed — one output beat per accepted input beat,
sustained at full rate when neither side stalls (see Verification plan's
full-throughput test).

**Amendment (see "Amendment" section below and `doc/canny_window3x3.md`
"Timing/latency" for the full, re-verified derivation):** the
implementation's actual latency is `g_img_width + 1` accepted input
beats, not `2*g_img_width + 2` — that was already flagged as a deliberate
deviation in "Implementation Notes (vhfill)" below. In addition, output
does **not** stay strictly 1:1 with `s_axis` for the whole frame: the
last `g_img_width + 1` output beats of every frame are produced during a
dedicated tail-flush ("draining") with no further input at all, since
gating output on genuinely fresh input (the fix for a separate bug, see
Amendment) means the frame's last already-buffered windows would
otherwise never be emitted.

## Corner cases

- FIFO depth vs. `g_img_width`: `fifo.fifo_wrapper`/`fifo.fifo` requires
  a *power-of-two* memory depth (`fifo.vhd`'s own elaboration-time
  assertion), which `g_img_width` need not be. Depth is chosen as
  `round_up_to_power_of_two(g_img_width + 1)` — strictly more than
  `g_img_width` even when `g_img_width` is itself already a power of
  two — so the line buffer never actually reaches true "full" capacity
  during normal write+simultaneous-read operation, sidestepping
  `fifo.vhd`'s documented one-cycle write-side pessimism ("write_ready
  looks at read_addr rather than read_addr_next ... there is a
  functional difference when the FIFO is full and a read performed makes
  the FIFO ready for another write ... low for one extra cycle"). Since
  *this* module's own `accepted_beats`-driven counters (not the FIFOs'
  own level/fullness signals) decide exactly when reads start, the extra
  headroom is pure safety margin, never load-bearing for correctness —
  see `doc/canny_window3x3.md` "Dependencies" for the exact citation.
- Out-of-frame taps at the 4 edges/corners: zero-masked in `m_axis_tdata`
  and excluded from the border OR-reduction (contribute `'0'`), per
  requirement — not "stale data happens to be zero," an explicit
  per-tap in-frame test drives this.
- `g_user_width = 1`: no incoming border bit exists on `s_axis_tuser` at
  all; internally forced to `'0'` for all 9 taps' border contribution
  (this instance only ORs in its own edge-of-frame test).
- Multiple back-to-back frames on the same instance: line buffers are
  never drained/reset between frames (only `rst_n` clears them); the
  `out_row`/`out_col` counters wrap naturally at `g_img_height`/
  `g_img_width` boundaries, so a second frame's first output beat lands
  correctly at `(0, 0)` without needing to re-incur the fill latency.

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" pattern, single shared `fire` signal driving
an internal fixed-depth delay-line pipeline (not a generic FSM) — see
"State machines" above.

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full AXI4-Stream backpressure on both sides. `m_axis_tvalid` must never
be combinationally dependent on `m_axis_tready` (correct per protocol:
`TVALID` must not wait for `TREADY`); `s_axis_tready` is gated by
`m_axis_tready` only once the window is actually producing valid output
(see "Architecture and dataflow" above) — before that, internal fill
advancement is independent of downstream readiness, since nothing is
offered downstream yet.

**Amendment:** the original plan above described `m_axis_tvalid` as "a
registered, sticky-once-set flag" — this turned out to be wrong and was
a real bug (see "Amendment" section below): a sticky `m_axis_tvalid`
stays asserted forever once primed, regardless of whether fresh input
actually arrived that cycle, causing phantom replayed output beats on
any input-side stall after priming. The corrected design computes
`window_valid` (`= m_axis_tvalid`) combinationally each cycle from
`s_axis_tvalid`, the (still sticky) `primed` flag, and a non-sticky
`draining` flag — `window_valid <= (s_axis_tvalid and primed) or
draining` — which does satisfy the stated "never depends on
`m_axis_tready`" rule (it depends on `s_axis_tvalid`, a different
channel's valid, not on `fire`, which would have reintroduced the
forbidden dependency via `s_axis_tready`). See `doc/canny_window3x3.md`
"Interfaces/protocols" and "Timing/latency" for the full explanation.

## Verification plan

- Dedicated VUnit unit test under `modules/canny_window3x3/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly (not
  the hdl-modules `bfm.*` wrappers — `s_axis_tuser` can be 1 bit, which
  fails `bfm.axi_stream_master`'s `user_width mod 8 = 0` assertion; see
  `shared/Vunit.md` §12's byte-alignment caveat).
- `stall_probability_percent` generic, swept via
  `module_canny_window3x3.py` (0 for the dedicated full-throughput test,
  nonzero otherwise), applied independently to the master and slave VC
  `stall_config`.
- `test_random_data`: small frame (`g_user_width=1`, no incoming border
  bit at all), random pixel data, per-beat expected window/border/tlast
  computed inline in the testbench (local 2D array model + the same
  per-tap in-frame masking formula as the DUT, independently re-derived
  from the requirement, not copy-pasted from the RTL).
- `test_full_throughput`: larger frame, zero stall on all sides,
  `check_relation` on total elapsed time vs. beat count + fill latency +
  small margin.
- `test_border_dilate`: `g_user_width=2`, small frame, deliberately
  nonzero incoming border bits at chosen positions (including interior
  points, to isolate the 9-tap OR-dilate from the self-edge test) plus
  the 4 frame corners/edges (to check the self-edge test and the
  data-tap zero-masking together).
- Prefer non-blocking `push_axi_stream`/`check_axi_stream(...,
  blocking => false)` per `shared/Vunit.md` §12.
- A Python-model file is not created separately (the transform is cheap
  enough to compute inline, following `shared/Vunit.md`'s example set by
  `axi_stream_join`'s own testbench for a structurally-similar case) —
  the inline VHDL model re-derives the masking/dilate formula from the
  requirement doc directly, not from the RTL under test.

## Implementation Notes (vhfill)

The `2*g_img_width + 2` accepted-beat priming threshold and the exact
per-tap in-frame masking formula are the two places most likely to need
an off-by-one fix against real simulation output; both are marked `--@`
in the backbone with a deliberately-wrong placeholder (see
`src/canny_window3x3.vhd`) so the RED testbench run fails for the
*expected* reason. Filled in below once `vhfill` completes:

As anticipated, both needed a fix, but the priming threshold needed more
than an "off-by-one" — the RED run caught a genuine latency-formula bug,
not just a fencepost slip:

- **In-frame masking**: filled in as anticipated, calling
  `tap_in_frame(out_row, out_col, row_off(row), col_off(col),
  g_img_width, g_img_height)` per tap.
- **Priming thresholds**: `mid_row_active`'s `g_img_width` threshold was
  already correct. `top_row_active` needed `2*g_img_width` (not
  `g_img_width`). `window_valid` needed `g_img_width + 1` — **not**
  `2*g_img_width + 2` as this document and `doc/canny_window3x3_req.md`
  state. That figure is provably wrong for `window_valid` given this
  design's ungated "bottom"/live-row tap chain (`bottom_mid_reg`/
  `bottom_left_reg` shift on every accepted beat from the first one, with
  no priming gate), which pins `window_valid`'s correct threshold to
  exactly `g_img_width + 1` — using `2*g_img_width + 2` instead would
  misalign the frozen `(out_row, out_col) = (0, 0)` against tap registers
  that have already advanced well past raster position `(0, 0)` by then.
  `2*g_img_width + 2` remains correct as the point at which the single
  most-delayed tap register (`top_left_reg`) first holds a real sample —
  just not as the `window_valid` threshold. See
  `doc/canny_window3x3.md` "Timing/latency" for the full derivation. This
  is flagged as a deviation from `doc/canny_window3x3_req.md` in the
  final task report, not silently changed there.
- **Also found during the RED confirmation run** (not one of the two
  `--@`-marked placeholders): a conditional signal assignment `border_in
  <= s_axis_tuser(1) when g_user_width = 2 else '0';` crashed at
  simulation start for the `g_user_width=1` instance with an out-of-bounds
  index — GHDL bounds-checks the literal index `1` against the actual
  elaborated port width even in the branch that is never selected at
  runtime. Fixed with a pair of `if ... generate` statements instead,
  since only the taken generate branch is ever elaborated.
- **Also found during GREEN**: the testbench's `check_axi_stream` calls
  omitted `blocking => false` (contrary to this document's own §"Prefer
  non-blocking ... blocking => false" guidance above), which is blocking
  by default in VUnit. Since `push_axi_stream`/`check_axi_stream` calls
  are interleaved per-beat in the same process, this deadlocked every
  test (2 ms watchdog timeout): the process blocked waiting for output
  beat 0 before ever pushing input beat 1, but the DUT needs
  `g_img_width + 1` accepted input beats before any output appears.
  Fixed by adding `blocking => false` to both `check_axi_stream` calls in
  `test/tb_canny_window3x3.vhd`.

## Amendment (later session — the `blocking => false` fix above had its own bug)

The `blocking => false` fix directly above was necessary but not
sufficient: the testbench never called `wait_until_idle` to drain the
non-blocking `check_axi_stream` queue before `test_runner_cleanup`, so
none of the queued checks ever actually ran before the simulation ended
— every test passed regardless of what the RTL did. Once that drain call
was added (a testbench-only fix, not part of this proposal's RTL) and
the checks started actually running, all three tests failed against the
RTL described above, exposing two further real, independent RTL bugs
that this proposal's design (and the initial `vhfill` pass above) had
missed, plus one build/tooling gotcha. Full derivation and cycle-by-cycle
reasoning: `doc/canny_window3x3.md` "Timing/latency". Summary:

- **Sticky `m_axis_tvalid` bug**: `m_axis_tvalid` was wired directly to
  the sticky priming flag (renamed `primed` in the fix; this proposal's
  "Architecture and dataflow" and "AXI4/AXI4-Stream protocol decisions"
  sections above called it `window_valid` and described it as itself the
  sticky flag — that description was the bug). Once primed, a sticky
  `m_axis_tvalid` never deasserts, so any `s_axis`-side stall after
  priming caused the same stale window to be re-offered as a phantom
  extra output transfer whenever `m_axis_tready` happened to also be
  high. Fix: `window_valid <= (s_axis_tvalid and primed) or draining` —
  a fresh, non-sticky signal computed every cycle, distinct from the
  still-sticky `primed` flag it depends on. Deliberately *not* `(fire and
  primed) or draining`, since `fire` depends on `s_axis_tready`, which
  itself depends on `m_axis_tready` once primed/draining — that would
  reintroduce the same-channel `m_axis_tvalid`-depends-on-`m_axis_tready`
  violation this proposal's "AXI4/AXI4-Stream protocol decisions" section
  correctly says must be avoided.
- **Missing tail-flush ("draining") bug**: gating output on genuinely
  fresh input (the fix above) has a consequence this proposal's
  "Architecture and dataflow"/"Latency/throughput" sections did not
  anticipate: the frame's last `g_img_width + 1` output windows have no
  further *new* input beat to ride along with, even though the data they
  need is already sitting in `fifo1`/`fifo2`. Without a dedicated
  mechanism to keep advancing after the last input beat is accepted,
  those windows would never be produced at all. Fix: two full-frame
  counters (`input_beat_count`/`output_beat_count`) derive
  `all_input_received`/`all_output_sent`/`draining`, and a shared `step`
  pulse (`step <= fire or (draining and m_axis_tready)`, deliberately
  tracking `fire` and *not* `window_valid`, to avoid desynchronizing the
  mid/top row FIFO-read pipeline by one beat) drives `out_row`/`out_col`
  advance and the mid/top row FIFO reads/shifts during both normal
  operation and draining.
- **GHDL `process(all)` sensitivity-inference gotcha** (tooling, not RTL
  logic): the `assemble_window` process read most of its inputs only
  indirectly, via a process-local `impure function` called from a nested
  `for` loop, rather than by name in the process body. GHDL 7.0.0-dev's
  `(all)` inference did not reliably pick this up: `m_axis_tdata` was
  computed once at elaboration and never recomputed, despite its real
  inputs changing every cycle. Fixed with an explicit sensitivity list.
  Not reproduced in isolated minimal repro attempts — the precise
  trigger condition in GHDL is not fully pinned down, but the fix is
  correct and safer regardless of mechanism.

Numeric thresholds are unchanged by this amendment: `mid_row_active` at
`g_img_width`, `top_row_active` at `2*g_img_width`, and `primed`
(formerly `window_valid`) at `g_img_width + 1` — re-verified
independently this time against a from-scratch cycle-accurate reference
model, not merely re-copied from the (untrustworthy) earlier derivation.
