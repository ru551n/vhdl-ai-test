# axi_stream_join

## Purpose

Generic 2-input AXI4-Stream rendezvous: joins two independent AXI4-Stream
links into one, emitting a joined transfer (data concatenated) only when
both inputs are simultaneously valid, and applying combined backpressure to
both inputs together. Not canny-specific — a general-purpose reusable
resynchronization primitive.

## Entity and architecture

Entity `axi_stream_join`, architecture `a`.

## Generics

| Name | Type | Default | Meaning | Constraints |
|---|---|---|---|---|
| `data_width_a` | positive | — | width of `s_axis_a_tdata` | > 0 |
| `data_width_b` | positive | — | width of `s_axis_b_tdata` | > 0 |

## Ports

| Name | Mode | Type/width | Description |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `s_axis_a_tvalid` | in | std_logic | input A valid |
| `s_axis_a_tready` | out | std_logic | backpressure to input A's producer |
| `s_axis_a_tdata` | in | std_logic_vector(data_width_a-1 downto 0) | input A payload |
| `s_axis_a_tuser` | in | std_logic_vector(1 downto 0) | bit0=SOF, bit1=border |
| `s_axis_a_tlast` | in | std_logic | end-of-line |
| `s_axis_b_tvalid` | in | std_logic | input B valid |
| `s_axis_b_tready` | out | std_logic | backpressure to input B's producer |
| `s_axis_b_tdata` | in | std_logic_vector(data_width_b-1 downto 0) | input B payload |
| `s_axis_b_tuser` | in | std_logic_vector(1 downto 0) | bit0=SOF, bit1=border |
| `s_axis_b_tlast` | in | std_logic | end-of-line |
| `m_axis_tvalid` | out | std_logic | joined output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(data_width_a+data_width_b-1 downto 0) | `s_axis_a_tdata` in upper bits, `s_axis_b_tdata` in lower bits |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) | bit0=SOF (from A), bit1=border (A or B) |
| `m_axis_tlast` | out | std_logic | end-of-line (from A) |

## Clocking and reset

Single clock (`clk`), resetless. The join is purely combinational
end-to-end, so no reset is needed.

## Interfaces/protocols

Three independent full AXI4-Stream elastic links (`s_axis_a`, `s_axis_b`,
`m_axis`), per `shared/Axi4.md`. Fires (all three transact together)
exactly when `s_axis_a_tvalid = '1'`, `s_axis_b_tvalid = '1'`, and
`m_axis_tready = '1'`, all in the same cycle. `m_axis_tvalid` is asserted
whenever both inputs are valid, independent of `m_axis_tready` (correct
per AXI4-Stream's `TVALID` rule); each input's `TREADY` is gated by the
other input's `TVALID` *and* `m_axis_tready`.

## Functional behavior

- `m_axis_tdata = s_axis_a_tdata & s_axis_b_tdata` (concatenation).
- `m_axis_tuser(1) = s_axis_a_tuser(1) or s_axis_b_tuser(1)`.
- `m_axis_tuser(0) = s_axis_a_tuser(0)`; `m_axis_tlast = s_axis_a_tlast`
  (structural passthrough from A — both forks are expected to already
  agree whenever a joined beat fires).
- A simulation-only assertion flags a "fork desynchronization" error if
  `s_axis_a_tuser(0) /= s_axis_b_tuser(0)` or `s_axis_a_tlast /=
  s_axis_b_tlast` on a fired beat.

## Timing/latency

Zero-latency, purely combinational. Throughput: one joined beat per cycle
under full throughput on all three sides; otherwise gated 1:1 by the join
condition with no internal buffering.

## Registers/configuration

None.

## Dependencies

- `common.handshake_merger` (hdl-modules, `modules/common/src/handshake_merger.vhd`),
  instantiated with `num_interfaces => 2`, reused unmodified for the 2-way
  valid/ready arbitration.

## Implementation notes

(filled in by `vhfill` once implemented — as-built decisions, e.g. the
concrete mechanism chosen for the simulation-only assertion.)

## Verification notes

- Only-one-input-valid must not produce a spurious output beat or drop the
  valid input.
- Full-throughput (both inputs valid, `m_axis_tready` held high) must
  produce one output beat per cycle with no bubbles.
- Randomized `stall_config` on all three AXI4-Stream sides is mandatory
  per `shared/Vunit.md` §12, plus one directed zero-stall case.
- A directed negative test intentionally desynchronizes `tuser(0)`/`tlast`
  between A and B and expects the consistency assertion to fire.
