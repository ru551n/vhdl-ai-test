# canny_sobel3x3 — requirement

## Responsibility
3x3 Sobel gradient on the smoothed-pixel window: L1 magnitude and a
4-sector direction classification, forked into two independent AXI4-Stream
masters (magnitude, direction) from one accepted input beat.

## Generics
None (fixed 8-bit input pixel width, fixed 11-bit magnitude width).

## Ports
| Port | Dir | Type |
|---|---|---|
| `clk` | in | std_logic |
| `rst_n` | in | std_logic |
| `s_axis_tvalid` | in | std_logic |
| `s_axis_tready` | out | std_logic |
| `s_axis_tdata` | in | std_logic_vector(71 downto 0) — smoothed-pixel window, same packing as `canny_gaussian3x3`'s input |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) — bit 0 = SOF, bit 1 = border |
| `s_axis_tlast` | in | std_logic — EOL |
| `m_axis_mag_tvalid` | out | std_logic |
| `m_axis_mag_tready` | in | std_logic |
| `m_axis_mag_tdata` | out | std_logic_vector(10 downto 0) (unsigned, 0..2040) |
| `m_axis_mag_tuser` | out | std_logic_vector(1 downto 0) — passthrough unchanged |
| `m_axis_mag_tlast` | out | std_logic — passthrough unchanged |
| `m_axis_dir_tvalid` | out | std_logic |
| `m_axis_dir_tready` | in | std_logic |
| `m_axis_dir_tdata` | out | std_logic_vector(1 downto 0) |
| `m_axis_dir_tuser` | out | std_logic_vector(1 downto 0) — passthrough unchanged, identical value to `m_axis_mag_tuser` |
| `m_axis_dir_tlast` | out | std_logic — passthrough unchanged, identical value to `m_axis_mag_tlast` |

## Protocols
Combinational Sobel + classify, registered one cycle (both fork outputs
driven from the same register so they present the same value/timing
relationship to their respective downstream consumers). The two forks
advance only when **both** consumers accept in the same cycle:
`s_axis_tready <= m_axis_mag_tready and m_axis_dir_tready`, and internally
`m_axis_mag_tvalid`/`m_axis_dir_tvalid` are asserted/deasserted together —
this avoids ever producing a magnitude beat without its matching direction
beat (or vice versa) being simultaneously produced, per `doc/canny_arch.md`
"Backpressure design" point 2. This AND-of-both-readies fork gating is the
one piece of this module not covered by a straight `handshake_pipeline`
reuse; a `handshake_splitter` (hdl-modules `common`) reuse was evaluated
for the fan-out and is the preferred implementation strategy at `vhfill`
time over a hand-written AND-gate, since it already implements the
sticky-per-output "already transacted" bookkeeping needed when the two
forks' consumers stall for different durations.

## Clock/reset
Single clock, synchronous active-low reset clears both output registers.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

- `Gx = (tr + 2*mr + br) - (tl + 2*ml + bl)` (signed, range -1020..1020),
  where `tl..br` are the 9 taps unpacked from `s_axis_tdata`.
- `Gy = (bl + 2*bm + br) - (tl + 2*tm + tr)` (signed, range -1020..1020).
- `m_axis_mag_tdata = |Gx| + |Gy|` (unsigned, max 2040, fits 11 bits).
- Direction sector (`m_axis_dir_tdata`), using `ax=|Gx|`, `ay=|Gy|`:
  - `"00"` (0°, horizontal gradient / vertical edge) if `ay <= (ax >> 1)`
  - `"10"` (90°, vertical gradient / horizontal edge) if `ax <= (ay >> 1)`
  - otherwise diagonal: `"01"` (45°) if `Gx` and `Gy` have the same sign
    (`sign(Gx) xor sign(Gy) = '0'`), else `"11"` (135°)
  - `ax<=(ay>>1)` is checked only when the `"00"` condition is false, so
    ties where both conditions could hold (only possible when `ax=ay=0`)
    resolve to `"00"`.
- When `s_axis_tuser(1)` (border) is `'1'`: `m_axis_mag_tdata` forced to 0,
  `m_axis_dir_tdata` forced to `"00"`; `m_axis_mag_tuser(1)` and
  `m_axis_dir_tuser(1)` both simply pass `s_axis_tuser(1)` through
  unchanged (this module does no windowing of its own and does not
  compute border itself).
