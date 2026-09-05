# canny_sobel3x3

## Purpose

3x3 Sobel gradient magnitude (L1 norm, 11-bit) and direction (4-sector,
2-bit) computed from one accepted smoothed-pixel window beat, forked into
two independent AXI4-Stream masters (`m_axis_mag`, `m_axis_dir`). See
`modules/canny/doc/canny_sobel3x3_req.md` and
`modules/canny/doc/canny_sobel3x3_proposal.md`.

## Entity and architecture

Entity `canny_sobel3x3`, architecture `a`.

## Generics

None (fixed 8-bit input pixel width, fixed 11-bit magnitude width).

## Ports

| Name | Mode | Type/width | Description |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `reset` | in | std_logic | synchronous active-high reset |
| `s_axis_tvalid` | in | std_logic | input valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(71 downto 0) | 3x3 smoothed-pixel window, same packing as `canny_gaussian3x3`'s input |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) | bit0=SOF, bit1=border |
| `s_axis_tlast` | in | std_logic | EOL |
| `m_axis_mag_tvalid` | out | std_logic | magnitude fork valid |
| `m_axis_mag_tready` | in | std_logic | backpressure from magnitude consumer |
| `m_axis_mag_tdata` | out | std_logic_vector(10 downto 0) | unsigned magnitude, 0..2040 |
| `m_axis_mag_tuser` | out | std_logic_vector(1 downto 0) | passthrough, identical to `m_axis_dir_tuser` |
| `m_axis_mag_tlast` | out | std_logic | passthrough, identical to `m_axis_dir_tlast` |
| `m_axis_dir_tvalid` | out | std_logic | direction fork valid |
| `m_axis_dir_tready` | in | std_logic | backpressure from direction consumer |
| `m_axis_dir_tdata` | out | std_logic_vector(1 downto 0) | 4-sector direction code |
| `m_axis_dir_tuser` | out | std_logic_vector(1 downto 0) | passthrough, identical to `m_axis_mag_tuser` |
| `m_axis_dir_tlast` | out | std_logic | passthrough, identical to `m_axis_mag_tlast` |

## Clocking and reset

Single clock (`clk`), synchronous active-high reset (`reset`). `reset`
clears the one hand-written `reg_valid`/`reg_data` register that both
forks are driven from, which combinationally forces both
`m_axis_mag_tvalid` and `m_axis_dir_tvalid` low the same cycle.

**Known limitation:** `common.handshake_splitter` (used for the fork,
see "Dependencies") exposes no reset port. A `reset` pulse landing at the
exact moment one fork has transacted the current beat and the other has
not can leave that fork's internal sticky bit set across the reset,
suppressing its valid for one beat after reset before the two forks'
sticky state self-resynchronizes on the next full join. Out of scope for
this module's verification (this project's testbenches only pulse
`reset` once before any traffic, per `tb_axi_stream_join`/
`tb_canny_window3x3` convention).

## Interfaces/protocols

Full AXI4-Stream elastic handshake on all three links. `s_axis_tready` is
the hand-written register's accept condition (never a combinational
function of `m_axis_mag_tvalid`/`m_axis_dir_tvalid`, and never a
combinational loop). `m_axis_mag_tvalid`/`m_axis_dir_tvalid` are
combinational functions of internal state only (`reg_valid` and
`handshake_splitter`'s own sticky bits), never waiting on
`m_axis_mag_tready`/`m_axis_dir_tready` to assert validity. Both fork
valids are always sourced from the same one-cycle-registered value
(`reg_data`), so their `tdata`/`tuser`/`tlast` can never independently
desync -- only their `tvalid` timing (via `handshake_splitter`'s
per-output sticky bookkeeping) can differ while one fork's consumer stalls
longer than the other's.

## Functional behavior

- `unpack_window` slices the 8 taps Sobel uses (`tl, tm, tr, ml, mr, bl,
  bm, br`) out of `s_axis_tdata`; the center tap `mm` (bits 39:32) is
  unused.
- `Gx = (tr + 2*mr + br) - (tl + 2*ml + bl)`, `Gy = (bl + 2*bm + br) -
  (tl + 2*tm + tr)` (signed, computed with 12-bit headroom; true range
  -1020..1020).
- `ax = |Gx|`, `ay = |Gy|` (unsigned, 11-bit).
- `mag = ax + ay` (unsigned, max 2040, fits 11 bits exactly).
- Direction: `"00"` if `ay <= (ax >> 1)`; else `"10"` if `ax <= (ay >>
  1)`; else `"01"` if `Gx`/`Gy` share a sign (`sign(Gx) xor sign(Gy) =
  '0'`, sign taken from each value's own MSB), else `"11"`. The `"00"`
  branch is checked first, so `ax=ay=0` resolves to `"00"`.
- Border (`s_axis_tuser(1)='1'`): `mag` forced to `0`, `dir` forced to
  `"00"`, applied after the raw calculation above; `tuser`/`tlast` always
  pass through unchanged regardless of border.
- The above combinational result is packed into a 16-bit word (`mag(11) &
  dir(2) & tuser(2) & tlast(1)`), registered once (`reg_data`, gated by
  `reg_valid`), then fanned out unmodified (just re-sliced) to both
  `m_axis_mag_t*` and `m_axis_dir_t*`.
- The fork's joint-acceptance/backpressure logic is entirely delegated to
  `common.handshake_splitter` (`num_interfaces => 2`) -- see "Dependencies".

## Timing/latency

One cycle of latency from an accepted `s_axis` beat to the corresponding
`m_axis_mag`/`m_axis_dir` beat. Throughput: 1:1 with `s_axis` once
running, sustained under backpressure on either fork independently
(`handshake_splitter`'s sticky bookkeeping is what makes this correct
rather than merely convenient -- see "Dependencies" and the proposal doc's
"Architecture and dataflow" section for the concrete bug a hand-written
AND-gate fork would have here).

## Registers/configuration

None (no run-time configuration registers; this module has no generics).

## Dependencies

- `common.handshake_splitter` (hdl-modules,
  `modules/common/src/handshake_splitter.vhd`), `num_interfaces => 2`,
  reused unmodified for the fork's joint-acceptance gating and per-output
  sticky "already transacted" bookkeeping. Pure control/handshake (no
  data ports); this module owns all data registration/multiplexing.
  Evaluated and preferred over a hand-written
  `s_axis_tready <= m_axis_mag_tready and m_axis_dir_tready` AND-gate
  specifically because the AND-gate variant duplicates a beat on whichever
  fork accepts first when the two forks stall for different durations
  (see proposal doc).
- `common.handshake_pipeline` was evaluated for the one-cycle register but
  **not used** -- it exposes no reset port, and this module's
  `reset` must clear the output register per the requirement (see
  proposal doc "Architecture and dataflow"). The register is instead
  hand-written (a single flip-flop stage using the same
  `input_ready <= output_ready or not output_valid` elastic-register
  logic `handshake_pipeline`'s own non-skid-buffer mode uses internally),
  with an added synchronous `reset` clear.

## Implementation notes

- `to_grad`: zero-extends an 8-bit unsigned tap to a 12-bit signed value
  (via `resize` then a `signed(...)` reinterpret cast -- safe because the
  zero-extended top bits guarantee a clear sign bit for any 0..255 input).
- `ax`/`ay` are sliced from `abs(Gx)`/`abs(Gy)` (which numeric_std returns
  at the same width as the signed input, 12 bits) down to 11 bits; safe
  because the true maximum magnitude (1020) never sets the discarded top
  bit.
- `mag_sum` is computed as a 12-bit unsigned addition of `ax`+`ay` (extra
  carry-safety headroom bit) before being truncated to the 11-bit
  `m_axis_mag_tdata` width; safe because the true maximum (2040) fits in
  11 bits.
- No other deviations from `doc/canny_sobel3x3_proposal.md`, beyond the
  documented `handshake_pipeline`-reset and `handshake_splitter`-reset-gap
  notes above (both already called out in the proposal's Implementation
  Notes).

## Verification notes

- Dedicated VUnit unit test under `modules/canny/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly for all
  three streams (2-bit `tuser` on all three fails
  `bfm.axi_stream_master/slave`'s byte-alignment assertion, same rationale
  as `canny_window3x3`).
- Independent `stall_probability_percent_mag`/`stall_probability_percent_dir`
  generics (plus one for the input side), swept via
  `module_canny.py`: an all-zero full-throughput config, plus
  asymmetric nonzero configs (e.g. 10 / 30) specifically to exercise the
  differently-stalled-forks case.
- Separate concurrent checking processes for `mag` and `dir`
  (non-blocking `check_axi_stream` each), so one fork's stall/backlog
  cannot block the other's checking process.
- Random-window correctness test with expected `Gx`/`Gy`/`mag`/`dir`
  computed inline per the exact requirement formulas.
- Directed case for each of the 4 direction sectors.
- Directed border-forces-zero case.
- Every-beat cross-check that `mag`/`dir` carry the same `tuser`/`tlast`.
- Full-throughput (zero stall) timing check via `check_relation`.
