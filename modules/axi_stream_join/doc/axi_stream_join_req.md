# axi_stream_join — requirement

## Responsibility
Generic 2-input AXI4-Stream rendezvous: emits a joined transfer (data
concatenated) only when both inputs are simultaneously valid, applying
combined backpressure to both inputs together. Not canny-specific — used
in this pipeline to resynchronize the Sobel magnitude/direction forks
before `canny_nms` without assuming equal per-branch latency, but has no
dependency on any canny-specific type or width.

## Generics
| Generic | Type | Notes |
|---|---|---|
| `data_width_a` | positive | width of `s_axis_a_tdata` |
| `data_width_b` | positive | width of `s_axis_b_tdata` |

## Ports
| Port | Dir | Type |
|---|---|---|
| `clk` | in | std_logic |
| `s_axis_a_tvalid` | in | std_logic |
| `s_axis_a_tready` | out | std_logic |
| `s_axis_a_tdata` | in | std_logic_vector(data_width_a-1 downto 0) |
| `s_axis_a_tuser` | in | std_logic_vector(1 downto 0) — bit 0 = SOF, bit 1 = border |
| `s_axis_a_tlast` | in | std_logic — EOL |
| `s_axis_b_tvalid` | in | std_logic |
| `s_axis_b_tready` | out | std_logic |
| `s_axis_b_tdata` | in | std_logic_vector(data_width_b-1 downto 0) |
| `s_axis_b_tuser` | in | std_logic_vector(1 downto 0) — bit 0 = SOF, bit 1 = border |
| `s_axis_b_tlast` | in | std_logic — EOL |
| `m_axis_tvalid` | out | std_logic |
| `m_axis_tready` | in | std_logic |
| `m_axis_tdata` | out | std_logic_vector(data_width_a+data_width_b-1 downto 0) — `s_axis_a_tdata` in the upper `data_width_a` bits, `s_axis_b_tdata` in the lower `data_width_b` bits |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) — see Functional Description |
| `m_axis_tlast` | out | std_logic — see Functional Description |

## Protocols
Thin wrapper around hdl-modules `common.handshake_merger` (reused
unmodified for the 2-way valid/ready arbitration) per
`shared/ReusableRTL.md` — `handshake_merger` is a pure combinational
function of the two `tvalid`s and `m_axis_tready` (fires only when both
inputs are valid; asserts both `tready`s together only on that same
cycle) and explicitly does not handle data itself, so this module adds:
data concatenation into `m_axis_tdata`, the `tuser(1)` OR-reduction, and
the `tuser(0)`/`tlast` passthrough-with-consistency-check described below.
No internal register/pipeline stage — combinational pass-through gated by
`handshake_merger`'s fire condition, matching `handshake_merger`'s own
zero-latency contract.

## Clock/reset
Single clock, resetless. No stateful elements besides what
`handshake_merger` itself requires (none, per its own datasheet — it is
purely combinational); the module is purely combinational end-to-end, so
no reset is needed.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

- `m_axis_tuser(1)` (border) = `s_axis_a_tuser(1) or s_axis_b_tuser(1)` —
  either fork's window/FIFO can independently mark border; if either says
  border, the joined result is border.
- `m_axis_tuser(0)` (SOF) and `m_axis_tlast` (EOL) are structurally taken
  from `s_axis_a`'s value (the two inputs are only ever joined at moments
  when both are valid, and by construction of the pipeline both forks
  originate from the same upstream `canny_sobel3x3` beat, so `s_axis_a`'s
  and `s_axis_b`'s SOF/EOL must already agree whenever `handshake_merger`
  fires). A simulation-only assertion checks
  `s_axis_a_tuser(0) = s_axis_b_tuser(0)` and `s_axis_a_tlast =
  s_axis_b_tlast` on every fired beat and reports a synchronization bug
  (not a border/data bug) if they ever disagree — this would indicate the
  two forks lost frame alignment somewhere upstream, which this module
  cannot itself recover from.
- This module is intentionally generic (no canny-specific port names or
  widths) so it can be reused for any future 2-way AXI4-Stream
  resynchronization need in this or another project.
