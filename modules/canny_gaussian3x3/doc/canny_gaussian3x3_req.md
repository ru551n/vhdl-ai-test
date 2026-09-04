# canny_gaussian3x3 — requirement

## Responsibility
3x3 approximate Gaussian smoothing of the raw pixel window, as a full
AXI4-Stream elastic stage (not windowing itself — consumes the window
produced by the upstream `canny_window3x3` instance).

## Generics
None (fixed 8-bit pixel width and fixed integer weights).

## Ports
| Port | Dir | Type |
|---|---|---|
| `clk` | in | std_logic |
| `rst_n` | in | std_logic |
| `s_axis_tvalid` | in | std_logic |
| `s_axis_tready` | out | std_logic |
| `s_axis_tdata` | in | std_logic_vector(71 downto 0) — 3x3 raw-pixel window, packed row-major MSB-to-LSB: `w_tl`=71:64, `w_tm`=63:56, `w_tr`=55:48, `w_ml`=47:40, `w_mm`=39:32, `w_mr`=31:24, `w_bl`=23:16, `w_bm`=15:8, `w_br`=7:0 |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) — bit 0 = SOF, bit 1 = border |
| `s_axis_tlast` | in | std_logic — EOL |
| `m_axis_tvalid` | out | std_logic |
| `m_axis_tready` | in | std_logic |
| `m_axis_tdata` | out | std_logic_vector(7 downto 0) — smoothed pixel |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) — passthrough unchanged |
| `m_axis_tlast` | out | std_logic — passthrough unchanged |

## Protocols
Combinational function of the 9 unpacked taps, wrapped in a single elastic
registered stage reused from hdl-modules' `common.handshake_pipeline`
(thin wrapper per `shared/ReusableRTL.md` — this module supplies the
combinational function and data width; `handshake_pipeline` supplies the
elastic register and `s_axis_tready`/`m_axis_tvalid` handshake logic).
`s_axis_tuser`/`s_axis_tlast` ride through the same `handshake_pipeline`
instance as extra data-bit lanes (not a second parallel pipeline), so they
can never desync from `m_axis_tdata`.

## Clock/reset
Single clock, synchronous active-low reset clears the `handshake_pipeline`
output register.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

`m_axis_tdata = (tl + 2*tm + tr + 2*ml + 4*mm + 2*mr + bl + 2*bm + br) / 16`,
using integer weights `[1,2,1;2,4,2;1,2,1]` (sum = 16, exact power-of-two
divide via right-shift, no rounding beyond truncation), where `tl..br` are
the 9 taps unpacked from `s_axis_tdata`. When `s_axis_tuser(1)` (border,
carried alongside the window) is `'1'`, `m_axis_tdata` is forced to `0`;
`m_axis_tuser(1)` simply passes `s_axis_tuser(1)` through unchanged — this
module does not compute border itself (it does no windowing of its own),
it only reacts to it by zeroing its numeric output. The value is otherwise
not meaningful at the border per the architecture's "growing border"
convention.
