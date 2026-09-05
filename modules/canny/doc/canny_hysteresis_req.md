# canny_hysteresis — requirement

## Responsibility
Local (3x3) hysteresis: a pixel is a final edge if it is classified
"strong", or classified "weak" and at least one of its 8 neighbors (in the
classification window) is "strong". See `doc/canny_arch.md` point 5 for
the explicit approximation vs. true global connected-component hysteresis.
This module drives the IP's top-level `m_axis` directly (last stage in the
pipeline — see block diagram), so its output port widths/format match the
top-level boundary exactly, not the internal 2-bit-`tuser` convention used
between other internal stages.

## Generics
None (fixed 2-bit classification width, fixed 8-bit output byte).

## Ports
| Port | Dir | Type |
|---|---|---|
| `clk` | in | std_logic |
| `reset` | in | std_logic |
| `s_axis_tvalid` | in | std_logic |
| `s_axis_tready` | out | std_logic |
| `s_axis_tdata` | in | std_logic_vector(17 downto 0) — 3x3 classification window, packed row-major MSB-to-LSB, 2 bits per tap: `w_tl`=17:16 ... `w_br`=1:0 |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) — bit 0 = SOF, bit 1 = border |
| `s_axis_tlast` | in | std_logic — EOL |
| `m_axis_tvalid` | out | std_logic |
| `m_axis_tready` | in | std_logic |
| `m_axis_tdata` | out | std_logic_vector(7 downto 0) — final edge byte (bit 0 = edge, bits 7:1 = '0'), matches `canny_top`'s `m_axis_tdata` exactly |
| `m_axis_tuser` | out | std_logic_vector(0 downto 0) — bit 0 = SOF only, passthrough of `s_axis_tuser(0)`. Border (`s_axis_tuser(1)`) is consumed here (forces `edge_out='0'`) but is **not** re-exposed — matches `canny_top`'s 1-bit `m_axis_tuser` exactly, per `doc/canny_arch.md` "Border signal carrying (addendum, rev 2.1)" |
| `m_axis_tlast` | out | std_logic — passthrough of `s_axis_tlast` unchanged |

## Protocols
Combinational compare, wrapped in a single elastic registered stage reused
from hdl-modules' `common.handshake_pipeline` (thin wrapper, same pattern
as `canny_gaussian3x3`), with the `tuser` width narrowed from 2 bits (in)
to 1 bit (out) inside the wrapper — the border bit is consumed by the
combinational function (to force `edge_out='0'`) rather than carried
through as a passenger bit, unlike every other internal stage in this
pipeline.

## Clock/reset
Single clock, synchronous active-high reset clears the `handshake_pipeline`
output register.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

The 3x3 classification window taps (`w_tl..w_br`) are unpacked from
`s_axis_tdata`. `edge = '1'` if `w_mm = "10"` (strong), or `w_mm = "01"`
(weak) and any of the other 8 taps `= "10"`. `s_axis_tuser(1)` (border)
forces `edge = '0'` regardless of the window comparison (border pixels are
never reported as edges, per the architecture's "growing border"
convention). `m_axis_tdata = "0000000" & edge` (bit 0 = edge, bits 7:1 =
`'0'`). `m_axis_tuser(0)` passes `s_axis_tuser(0)` (SOF) through
unchanged; there is no `m_axis_tuser(1)` — border is fully consumed at
this stage and not propagated further, since this module's output feeds
the top-level `m_axis` directly.
