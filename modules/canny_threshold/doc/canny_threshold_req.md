# canny_threshold — requirement

## Responsibility
Double-threshold classification of the suppressed magnitude into
none/weak/strong.

## Generics
| Generic | Type | Notes |
|---|---|---|
| `g_thresh_low` | natural | below this: "none" |
| `g_thresh_high` | natural | at/above this: "strong"; between: "weak" |

## Ports
| Port | Dir | Type |
|---|---|---|
| `clk` | in | std_logic |
| `rst_n` | in | std_logic |
| `s_axis_tvalid` | in | std_logic |
| `s_axis_tready` | out | std_logic |
| `s_axis_tdata` | in | std_logic_vector(10 downto 0) — suppressed magnitude |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) — bit 0 = SOF, bit 1 = border |
| `s_axis_tlast` | in | std_logic — EOL |
| `m_axis_tvalid` | out | std_logic |
| `m_axis_tready` | in | std_logic |
| `m_axis_tdata` | out | std_logic_vector(1 downto 0) ("00"=none,"01"=weak,"10"=strong) |
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

`m_axis_tdata = "10"` if `s_axis_tdata >= g_thresh_high`; `"01"` if
`g_thresh_low <= s_axis_tdata < g_thresh_high`; else `"00"`.
`g_thresh_low <= g_thresh_high` is assumed (an elaboration-time assertion
checks it). When `s_axis_tuser(1)` (border) is `'1'`, `m_axis_tdata`
forced to `"00"`; `m_axis_tuser(1)` simply passes `s_axis_tuser(1)`
through unchanged (this module does no windowing of its own).
