# canny_window3x3

## Purpose

Generic reusable AXI4-Stream 3x3 sliding-window generator over a continuous
raster-scan stream of `img_width x img_height` samples, with full elastic
backpressure (`TREADY`) on both sides. Reused by four canny pipeline stages
(W1 raw, W2 smoothed, W3 magnitude, W4 classification) with different
`data_width`. See `modules/canny/doc/canny_window3x3_req.md` and
`modules/canny/doc/canny_window3x3_proposal.md`.

## Entity and architecture

Entity `canny_window3x3`, architecture `a`.

## Generics

| Name | Type | Meaning | Constraints |
|---|---|---|---|
| `img_width` | positive | frame width | > 0 |
| `img_height` | positive | frame height | > 0 |
| `data_width` | positive | per-sample bit width | > 0 |
| `user_width` | positive range 1 to 2 | width of `s_axis_tuser`; `1` = no incoming border bit (forced `'0'` internally), `2` = bit0 SOF, bit1 border | 1 or 2 |

## Ports

| Name | Mode | Type/width | Description |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `reset` | in | std_logic | synchronous active-high reset |
| `s_axis_tvalid` | in | std_logic | input valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(data_width-1 downto 0) | input sample |
| `s_axis_tuser` | in | std_logic_vector(user_width-1 downto 0) | bit0=SOF, bit1 (only if width=2)=border |
| `s_axis_tlast` | in | std_logic | end-of-line |
| `m_axis_tvalid` | out | std_logic | output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(9*data_width-1 downto 0) | 3x3 window, row-major MSB-to-LSB: `w_tl` top bits ... `w_br` bottom bits |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) | bit0=SOF (passthrough), bit1=border (recomputed dilate) |
| `m_axis_tlast` | out | std_logic | EOL, recomputed from this module's own output-side column counter |

## Clocking and reset

Single clock (`clk`), synchronous active-high reset (`reset`). `reset` clears
this module's own bookkeeping: `accepted_beats` priming counter, `out_row`/
`out_col` output-position counters, and the column-tap shift registers.

**Known limitation:** the two internal line-buffer FIFOs (reused
`fifo.fifo_wrapper` instances) are **not** cleared by `reset`, because that
hdl-modules primitive exposes no reset port at all (confirmed against
`hdl-modules:modules/fifo/src/fifo_wrapper.vhd`/`fifo.vhd`). At elaboration
their internal address pointers start at their VHDL default (zero, i.e.
empty), so a `reset` pulse applied once before any beat is ever accepted
(the pattern used by every existing testbench in this repo, e.g.
`tb_axi_stream_join`'s `reset_gen`) behaves correctly. A *mid-stream* reset
would leave stale pre-reset samples queued in the FIFOs ahead of
post-reset writes; that scenario is out of scope for this module (no
verification test exercises it) and would need an external drain/flush
mechanism to support, which reusing this primitive cannot itself provide.

## Interfaces/protocols

Full AXI4-Stream elastic handshake on both sides per `shared/Axi4.md`.
`fire = s_axis_tvalid and s_axis_tready` gates the live/bottom row's shift
and every accepted-beat counter; `step` (`fire`, plus a drain-only pulse --
see "Timing/latency" below) gates the mid/top row FIFO reads and column-tap
shifts, so a stall on `m_axis_tready` (once primed) stops the whole
pipeline from advancing that cycle -- no bubble insertion, no re-fetch, no
dropped/duplicated sample.

**`m_axis_tvalid` (`window_valid`) is never combinationally dependent on
`m_axis_tready`** -- this is a correctness requirement, not just a style
preference: `s_axis_tready`/`down_ready` *does* depend on `m_axis_tready`
(once primed or draining -- see "Timing/latency"), so if `window_valid` were
built from `fire` (`= s_axis_tvalid and s_axis_tready`) instead of
`s_axis_tvalid` directly, `m_axis_tvalid` would transitively depend on
`m_axis_tready` -- a same-channel VALID-depends-on-READY combinational path
that AXI4-Stream forbids and that can deadlock against a downstream sink
whose own `TREADY` depends on this module's `TVALID` (a common, otherwise
unremarkable sink pattern). `window_valid <= (s_axis_tvalid and primed) or
draining` avoids this: it depends on `s_axis_tvalid` (a different channel's
valid) and on plain internal state (`primed`/`draining`), never on
`m_axis_tready`. `s_axis_tready` is gated by `m_axis_tready` only once the
window is actually producing valid output (once `primed` or `draining`) --
that cross-channel dependency (input-side readiness depending on
output-side readiness) is the normal, correct elastic-pipeline backpressure
pattern and is not the same thing as the same-channel violation above.

## Functional behavior

- Two elastic line-buffer FIFOs (`fifo.fifo_wrapper`, synchronous mode) each
  delay a live raster row by `img_width` accepted beats; a 3-tap
  ("right"=live/combinational, "mid"=1-cycle registered, "left"=2-cycle
  registered) column shift register per row-lane derives the 3
  column-neighbors of that row. See `doc/canny_window3x3_proposal.md`
  "Architecture and dataflow" for the full dataflow diagram.
- The window *may* become valid once `img_width + 1` accepted input beats
  have occurred (the `primed` threshold); output is then 1:1 with accepted
  input beats while fresh input keeps arriving, plus a `img_width + 1`
  -beat tail-flush ("draining") of already-buffered windows once the frame's
  last input beat has been accepted. **Deviation from
  `doc/canny_window3x3_req.md`'s stated `2*img_width + 2`** -- see
  "Implementation notes" below for why that figure does not hold for this
  architecture (an ungated live/bottom-row tap chain), and why
  `img_width + 1` is the value this module actually needs for
  `out_row`/`out_col` to stay correctly aligned with the windowed data. See
  "Timing/latency" below for the full `primed`/`draining`/`window_valid`
  distinction -- `primed` alone is *not* `m_axis_tvalid`.
- `m_axis_tuser(0)` = the center tap's (mid-row, mid-column) carried SOF bit
  (passthrough, delay-matched to the data through the same elastic lanes).
- `m_axis_tuser(1)` = this module's own edge-of-frame test on
  `(out_row, out_col)` **OR**'d with a 9-way OR-reduction of the incoming
  border bit carried alongside each of the 9 taps (each taken as `'0'` for
  any tap that is out-of-frame relative to `(out_row, out_col)`, and always
  `'0'` when `user_width=1`) -- the "growing border" dilate, see
  `doc/canny_arch.md` "Growing border (corrected, rev 2.2)".
- `m_axis_tdata`'s 9 `data_width`-wide slices are each tap's real stored
  sample when in-frame, else `(others => '0')`.
- `m_axis_tlast <= to_sl(out_col = img_width - 1)` -- recomputed, not
  passed through.

## Timing/latency

> **This section was re-derived from scratch and re-verified against an
> independent cycle-accurate model and the real RTL/waveforms.** An earlier
> revision of this section was written and "verified" against a testbench
> that had its own bug (VUnit `check_axi_stream(..., blocking => false)`
> calls were never drained with `wait_until_idle` before
> `test_runner_cleanup`, so none of the queued checks actually ran before
> the simulation ended -- the testbench always reported a pass regardless of
> what the RTL did). Once that testbench bug was fixed (elsewhere, not as
> part of this derivation) and the checks started actually running, all
> three tests failed against the RTL as it stood at the time, including
> `m_axis_tdata` containing undriven (`'U'`) bits on real output beats. The
> numeric per-tap fencepost thresholds below turned out to be unchanged from
> the earlier (untrustworthy) derivation, but two real, independent RTL bugs
> were found and fixed on top of them: (1) `m_axis_tvalid` was wired
> directly to the sticky priming threshold instead of also requiring fresh
> input, causing phantom replayed output beats on any stall once primed,
> and (2) there was no tail-flush ("draining") mechanism at all, so the
> last `img_width + 1` output windows of every frame -- already fully
> buffered and ready -- were never produced. See "New gotchas"/final report
> for the full incident writeup.

Latency: `img_width + 1` accepted input beats before the first valid
output beat; one-time per instance (line buffers are never drained between
frames, so subsequent frames on the same continuous stream do not re-incur
this fill delay). Throughput: 1:1 with `s_axis` once primed and while fresh
input keeps arriving, plus a `img_width + 1`-beat tail-flush at the end of
every frame (see "`window_valid` vs. `primed`" below) during which output
continues with no further input at all.

### Per-tap fencepost derivation (the `primed` threshold)

0-indexed accepted-beat count `t`, `W = img_width`: the "bottom"/live
row's column-tap registers (`bottom_mid_reg`/`bottom_left_reg`) shift
unconditionally on every accepted beat starting from the very first one (no
priming gate on that row), so at beat `t` they hold input sample `S(t-1)` /
`S(t-2)` (raster index, row-major) respectively, independent of any priming
decision. This pins down every other threshold:
- `mid_row_active` (fifo1 starts being read) must turn on at `t = W`, the
  instant row 1 starts arriving -- confirmed by matching the resulting
  `mid_mid_reg(t) = S(t-W-1)` against the required raster index
  `out_row*W + out_col` for every `t`.
- `top_row_active` (fifo2 starts being read) must turn on at `t = 2*W`, by
  the same argument one row-buffer-depth later.
- `primed` must turn on at `t = W + 1`: this is forced (not a free design
  choice) by requiring `bottom_mid_reg(t) = S(out_row+1, out_col)` to hold
  at the instant `(out_row, out_col)` first reads back `(0, 0)`. Turning it
  on any later (e.g. the naive "2 full row-buffers plus 2 taps" estimate of
  `2*W + 2`) would leave `out_row`/`out_col` frozen at `(0, 0)` while every
  tap register has already raced ahead to a later raster position (they are
  gated only by `mid_row_active`/`top_row_active`, not by `primed`),
  corrupting the very first output(s). `2*W + 2` is, instead, the beat
  count at which the single most-delayed register (`top_left_reg`) first
  holds a real (non-default-reset) sample -- an internal detail with no
  bearing on `primed`, since any tap that is out-of-frame relative to the
  currently emitted `(out_row, out_col)` is zero-masked regardless of
  whether its backing register happens to hold real or stale data yet.

These three numeric thresholds (`W`, `2*W`, `W + 1`) are unchanged from the
earlier (untrustworthy) derivation -- re-verified independently this time
against a from-scratch cycle-accurate reference model swept over many
width/height/seed/stall-pattern combinations, not just re-copied.

### `window_valid` vs. `primed`: the sticky-flag bug and its fix

**`primed` is a threshold flag, not `m_axis_tvalid`.** `primed` is derived
from a saturating, registered `accepted_beats` counter (`primed <=
accepted_beats >= img_width + 1`), so it is *sticky*: once true, it never
deasserts again for the rest of the frame. An earlier revision of this
module wired `m_axis_tvalid` directly to that sticky flag. That is a real,
pervasive correctness bug (not just a tail/drain issue): once primed,
`m_axis_tvalid` stayed asserted on *every* cycle regardless of whether a
fresh input beat had actually been accepted that cycle -- e.g. any
`s_axis`-side stall (gap in `s_axis_tvalid`) after priming -- silently
replaying the same stale `(out_row, out_col)` window as a phantom extra
output transfer whenever `m_axis_tready` happened to also be high that
cycle.

The fix: `window_valid <= (s_axis_tvalid and primed) or draining` --
`m_axis_tvalid` requires *fresh* upstream data (`s_axis_tvalid`), not merely
"primed", except while draining (see below, where by construction there is
no fresh upstream data left at all). Deliberately **not** `(fire and
primed) or draining` (`fire = s_axis_tvalid and s_axis_tready`) -- see
"Interfaces/protocols" above for why that would reintroduce a forbidden
same-channel `m_axis_tvalid`-depends-on-`m_axis_tready` combinational path.

### Tail-flush ("draining"): the missing-output bug and its fix

Even with the sticky-flag bug above fixed, gating everything on a *fresh*
input beat has a consequence: `out_row`/`out_col` only ever advance
alongside a genuine new `fire`. But the very last `img_width + 1` output
windows of a frame have no further *new* input beat to ride along with --
their only missing ingredient (the buffered mid/top row data) is already
sitting in `fifo1`/`fifo2`, it just has not been read out yet. Without a
dedicated mechanism to keep advancing after the frame's last input beat has
been accepted, those last `img_width + 1` output windows would never be
produced at all (this was a second, independent bug from the sticky-flag
one above, not a variant of it).

The fix adds two full-frame saturating counters, `input_beat_count` and
`output_beat_count` (range `0 to img_width * img_height`, distinct from
the small, fast-saturating `accepted_beats` used for the `primed`/
`*_row_active` thresholds above, which saturates far too early to detect
end-of-frame for any non-trivial image height):
- `all_input_received <= input_beat_count >= img_width * img_height`
- `all_output_sent <= output_beat_count >= img_width * img_height`
- `draining <= all_input_received and not all_output_sent`

and a single shared `step` pulse -- "a real output beat is being consumed
this cycle", used for `out_row`/`out_col` advance, the mid/top row FIFO
reads, and their column-tap shifts:
`step <= fire or (draining and m_axis_tready)`.

`step` must track `fire` directly (**not** `window_valid`/`fire and
primed`): `mid_row_active`/`top_row_active` turn on strictly *before*
`primed` does (`W`/`2*W` vs. `W + 1`), so the mid/top row's
fifo-read-and-shift pipeline must already be advancing on plain `fire`
during that pre-primed-but-row-active window, one beat ahead of `primed`
turning on. Gating it on `window_valid` instead would silently drop that
one beat's worth of `fifo1`/`fifo2` reads, permanently desynchronizing the
mid/top rows by one position for the rest of the frame. During draining
(tail flush, no fresh `fire` left since all input is already accepted),
the same pulse instead advances purely on `m_axis_tready`.

### GHDL `process(all)` sensitivity-inference gotcha (build/tooling, not RTL logic)

Independently of the two logic bugs above, the `assemble_window` process
(which builds `m_axis_tdata`/`m_axis_tuser(1)` from the tap registers) was
written as `process(all)`, reading most of its inputs only *indirectly* --
via a process-local `impure function tap_lane(row, col)` called from inside
a nested `for row loop / for col loop`, rather than by name directly in the
process body. Empirically, GHDL 7.0.0-dev's `(all)` sensitivity-list
inference does not reliably pick up signals read only that way: `raw VCD`
inspection showed `m_axis_tdata` computed exactly once (at
elaboration/reset) and never recomputed for the rest of the simulation,
despite the tap registers it reads genuinely changing every cycle -- so
`m_axis_tdata` kept reporting stale/garbage (in the observed failure,
still-`'U'`) data even on cycles where `m_axis_tvalid` was correctly
asserted and every signal it should depend on already held the correct
value. The fix: `assemble_window` now uses an explicit sensitivity list
(`out_row, out_col`, and all nine `*_live`/`*_mid_reg`/`*_left_reg` tap
signals) instead of `(all)`.

## Registers/configuration

None (no run-time configuration registers; all parameterization is via
generics).

## Dependencies

- `fifo.fifo_wrapper` (hdl-modules, `modules/fifo/src/fifo_wrapper.vhd`),
  instantiated twice in synchronous mode (`use_asynchronous_fifo => false`),
  as the two line-buffer FIFOs. Depth is
  `round_up_to_power_of_two(img_width + 1)` since `fifo.fifo_wrapper`'s
  synchronous-mode branch (`fifo.vhd`) asserts `is_power_of_two` on its
  memory depth at elaboration, and the `+1` gives one word of headroom so
  the FIFO's own occupancy never reaches its documented one-cycle
  write-side pessimism corner case (this module's own `accepted_beats`
  counter, not the FIFOs' level/fullness outputs, decides exactly when
  reads start, so the headroom is pure margin, never load-bearing).
- `math.math_pkg.round_up_to_power_of_two` (hdl-modules,
  `modules/math/src/math_pkg.vhd`) for the FIFO depth calculation above.

## Implementation notes

- Priming thresholds: `mid_row_active <= accepted_beats >= img_width`,
  `top_row_active <= accepted_beats >= 2*img_width`, `window_valid <=
  accepted_beats >= img_width + 1`. See "Timing/latency" above for the
  full derivation and the explicit, deliberate deviation from
  `doc/canny_window3x3_req.md`'s stated `2*img_width + 2` window-valid
  latency figure (that figure is provably the point at which the deepest
  tap register, `top_left_reg`, first holds a real sample -- not the
  point at which `window_valid`/`m_axis_tvalid` may correctly assert;
  using it for `window_valid` while `bottom_mid_reg`/`bottom_left_reg`
  shift unconditionally from the first accepted beat would misalign
  `out_row`/`out_col` against the already-advanced tap registers).
- In-frame masking (`assemble_window` process): each of the 9 taps calls
  `tap_in_frame(out_row, out_col, row_off(row), col_off(col),
  img_width, img_height)` to decide `m_axis_tdata`'s per-tap zero-mask
  and the tap's contribution (or not) to the border OR-reduction, per
  requirement.
- `border_in` (the incoming border bit, used only when `user_width = 2`)
  is derived via a pair of `if ... generate` statements
  (`gen_border_uw2`/`gen_border_uw1`), not a conditional signal assignment
  (`... when user_width = 2 else '0'`) -- see "New gotchas" in the final
  task report; GHDL bounds-checks a literal out-of-range index
  (`s_axis_tuser(1)`) against the actual elaborated port width even in a
  branch that is never selected at runtime, so a `when/else` on
  `user_width` crashes the `user_width=1` instance at simulation
  start. `generate` sidesteps this since only the taken branch is ever
  elaborated.
- No other deviations from `doc/canny_window3x3_proposal.md`.

## Verification notes

- Dedicated VUnit unit test under `modules/canny/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly (not the
  hdl-modules `bfm.*` wrappers, since `s_axis_tuser` can be 1 bit wide,
  which fails `bfm.axi_stream_master`'s `user_width mod 8 = 0` assertion).
- `stall_probability_percent` generic swept via `module_canny.py`
  (0 for the dedicated full-throughput test, nonzero otherwise).
- Random-data correctness test with per-beat expected window/border/tlast
  computed inline in the testbench.
- Full-throughput (zero stall) timing check via `check_relation`.
- Border-dilate test with `user_width=2` and deliberately nonzero
  incoming border bits, exercising all 4 frame edges/corners plus interior
  points to isolate the 9-tap OR-dilate from the self-edge test.
