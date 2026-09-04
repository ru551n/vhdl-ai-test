# canny_nms — requirement

## Responsibility
Non-maximum suppression: keep the center magnitude only if it is a local
maximum along the gradient direction; else suppress to 0. Consumes the
single joined AXI4-Stream produced by `axi_stream_join` (windowed
magnitude + direction resynchronized), not two separate streams itself.

## Generics
None (fixed 11-bit magnitude width).

## Ports
| Port | Dir | Type |
|---|---|---|
| `clk` | in | std_logic |
| `rst_n` | in | std_logic |
| `s_axis_tvalid` | in | std_logic |
| `s_axis_tready` | out | std_logic |
| `s_axis_tdata` | in | std_logic_vector(100 downto 0) — `axi_stream_join`'s concatenated output: bits 100:2 = 3x3 magnitude window (same 9x11 row-major packing as `canny_window3x3`'s output), bits 1:0 = direction sector |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) — bit 0 = SOF, bit 1 = border (the OR of both forks' border bits, computed by `axi_stream_join`) |
| `s_axis_tlast` | in | std_logic — EOL |
| `m_axis_tvalid` | out | std_logic |
| `m_axis_tready` | in | std_logic |
| `m_axis_tdata` | out | std_logic_vector(10 downto 0) — suppressed magnitude |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) — passthrough unchanged |
| `m_axis_tlast` | out | std_logic — passthrough unchanged |

## Protocols
Combinational compare, wrapped in a single elastic registered stage reused
from hdl-modules' `common.handshake_pipeline` (thin wrapper, same pattern
as `canny_gaussian3x3`). `s_axis_tuser`/`s_axis_tlast` ride through the
same instance as extra data-bit lanes.

## Clock/reset
Single clock, synchronous active-low reset clears the `handshake_pipeline`
output register.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

`direction_in = s_axis_tdata(1 downto 0)`; the 3x3 magnitude window taps
(`w_tl..w_br`) are unpacked from `s_axis_tdata(100 downto 2)`. Neighbor
pair selected by `direction_in`:
- `"00"` (0°): compare `w_mm` against `w_ml`, `w_mr` (west/east)
- `"01"` (45°): compare `w_mm` against `w_tr`, `w_bl` (NE/SW)
- `"10"` (90°): compare `w_mm` against `w_tm`, `w_bm` (north/south)
- `"11"` (135°): compare `w_mm` against `w_tl`, `w_br` (NW/SE)

`m_axis_tdata = w_mm` if `w_mm >= neighbor_a and w_mm >= neighbor_b`,
else `0` (ties resolve to "kept", i.e. non-strict `>=`, a documented
simplification that can retain double-width edges on perfectly flat
ridges). When `s_axis_tuser(1)` (border) is `'1'`, `m_axis_tdata` forced
to 0; `m_axis_tuser(1)` simply passes `s_axis_tuser(1)` through unchanged
(this module does no windowing of its own).
