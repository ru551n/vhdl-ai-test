# cnn_accel_pool

## Purpose

Spatial reduction over one pooling window per beat, produced by
`cnn_accel_window_gen`. `OPCODE_POOL_MAX` computes the int8 maximum of the
window's active taps and emits it directly (already a valid int8 result,
no further scaling). `OPCODE_POOL_AVG` computes the int32 sum of the
window's active taps and emits it for `cnn_accel_bias_requant` to
scale/round into int8 — dividing by the pool area is exactly a requantize
operation, performed downstream, not in this module (see
`doc/cnn_accel_arch.md` "Non-obvious boundary rationale"). The two outputs
are mutually exclusive per instruction, selected by `cfg_opcode`.

## Entity and architecture

Entity `cnn_accel_pool`, architecture `a`.

## Generics

| Generic | Type | Meaning / constraints |
|---|---|---|
| `g_max_kernel_size` | `positive` | Upper bound on `cfg_pool_kernel_h`/`cfg_pool_kernel_w` individually; sizes the fixed `g_max_kernel_size**2`-lane tap array. **Contract: `g_max_kernel_size**2 * 8 <= 128`** (`axi_stream_pkg.axi_stream_data_sz`, a fixed package constant, not a generic) — so `g_max_kernel_size <= 4` for any instantiation of this module. Checked by an elaboration-time `assert`. |
| `g_accum_width` | `positive` | `OPCODE_POOL_AVG` sum width. Must be wide enough to hold `g_max_kernel_size**2 * 127` without overflow (no saturation/rounding applied here). **Contract: `g_accum_width <= 128`**, checked by an elaboration-time `assert`. |

## Ports

| Port | Dir | Type | Description |
|---|---|---|---|
| `clk` | in | `std_ulogic` | Clock. |
| `reset` | in | `std_ulogic` | Synchronous active-high reset (`reset_internal` at the IP top level). Default `'0'`. |
| `cfg_opcode` | in | `std_ulogic_vector(7 downto 0)` | `OPCODE_POOL_MAX` vs `OPCODE_POOL_AVG` (`cnn_accel_pkg`), sampled at `s_window` accept time. Any value other than `OPCODE_POOL_AVG` is treated as the `OPCODE_POOL_MAX` path. |
| `cfg_pool_kernel_h`, `cfg_pool_kernel_w` | in | `std_ulogic_vector(7 downto 0)` | Pool kernel height/width for the in-flight instruction. Contract: each in `1 .. g_max_kernel_size`. `cfg_pool_kernel_h * cfg_pool_kernel_w` taps are active. |
| `s_window_m2s` / `s_window_s2m` | in / out | `axi_stream_pkg.axi_stream_m2s_t` / `axi_stream_s2m_t` | One pooling window per beat, from `cnn_accel_window_gen`. `data` low `cfg_pool_kernel_h * cfg_pool_kernel_w * 8` bits hold that many signed int8 taps: tap `i = row * cfg_pool_kernel_w + col` at bits `8*i + 7 downto 8*i` (row-major, ascending from the low bits). Remaining high bits are don't-care. |
| `m_max_m2s` / `m_max_s2m` | out / in | `axi_stream_pkg.axi_stream_m2s_t` / `axi_stream_s2m_t` | `OPCODE_POOL_MAX` result: int8 max, `data(7 downto 0)` (high bits `0`). To the final output `handshake_mux` (bypasses `cnn_accel_bias_requant`). |
| `m_avgsum_m2s` / `m_avgsum_s2m` | out / in | `axi_stream_pkg.axi_stream_m2s_t` / `axi_stream_s2m_t` | `OPCODE_POOL_AVG` result: `g_accum_width`-bit sum, `data(g_accum_width - 1 downto 0)` (high bits `0`). To `cnn_accel_bias_requant`. |

## Clocking and reset

Single clock domain (`clk`). Synchronous, active-high `reset`. Only the
pipeline's `valid` bit (`out_valid_q`) is reset; the held
result/tag/`last` registers are not (their content is irrelevant while
`out_valid_q='0'` and is fully overwritten together on the next accepted
beat).

## Interfaces/protocols

Three independent AXI4-Stream links (`s_window` in; `m_max`, `m_avgsum`
out), full backpressure on all three per `shared/Axi4.md`. `m_max_m2s.valid`
and `m_avgsum_m2s.valid` are structurally mutually exclusive (driven from
one shared tagged register) — never both `'1'` in the same cycle.
`s_window_s2m.ready` depends only on the currently-selected output's
`ready` (the other, unselected output's handshake is irrelevant to
accepting the next `s_window` beat).

## Functional behavior

On each accepted `s_window` beat: extract the active taps (per the tap-
packing convention above), reduce to a max (`OPCODE_POOL_MAX`) or sum
(`OPCODE_POOL_AVG`) combinationally, and register the result plus a
route tag and `last` into a one-entry output register. That register
drives `m_max_m2s`/`m_avgsum_m2s` (whichever the tag selects) until the
selected downstream accepts it, at which point a new `s_window` beat may
be accepted (registered-ready, one-entry, full-throughput elastic stage).

## Timing/latency

Fixed 1-cycle latency from an accepted `s_window` beat to the
corresponding output beat. Full throughput once started: one output beat
per accepted input beat, sustained indefinitely while the selected output
stays continuously ready.

## Registers/configuration

None (no CSR-mapped registers in this module; `cfg_*` inputs are driven
combinationally by `cnn_accel_layer_ctrl`).

## Dependencies

- `ieee.std_logic_1164`, `ieee.numeric_std`.
- `axi_stream.axi_stream_pkg` (`hdl-modules/modules/axi_stream/src/axi_stream_pkg.vhd`) —
  `axi_stream_m2s_t`/`axi_stream_s2m_t` record types, `axi_stream_data_sz`
  constant.
- `cnn_accel.cnn_accel_pkg` (`modules/cnn_accel/src/cnn_accel_pkg.vhd`) —
  `OPCODE_POOL_AVG` constant.

## Implementation notes

- Both reductions (`reduce_max`/`reduce_sum`) are written as an unrolled
  linear comparison/adder chain over the fixed `g_max_kernel_size**2`-lane
  tap array (masked by `active_count`, not by an identity-value
  substitution), not a literal balanced binary tree — functionally
  identical result/latency for this generic-bounded lane count
  (`g_max_kernel_size <= 4`, i.e. at most 16 lanes).
- One shared one-entry output register (not two independent per-port
  registers) — see `doc/cnn_accel_pool_proposal.md` section 4 for why this
  structurally guarantees output mutual exclusion instead of relying on a
  separate mux-select check.
- The exact tap-packing convention above (`i = row * cfg_pool_kernel_w +
  col`, ascending from the low bits) is this module's own choice, recorded
  since `cnn_accel_window_gen` did not exist yet at design time — the
  `cnn_accel_window_gen` implementation must match it.

## Verification notes

See `modules/cnn_accel/test/tb_cnn_accel_pool.vhd`
(`tb_cnn_accel_pool`/architecture `tb`) and
`doc/cnn_accel_pool_proposal.md`'s "Verification plan" for the full test
list. Key corner cases covered: kernel sizes from `1x1` up to
`g_max_kernel_size x g_max_kernel_size` (square and non-square), int8
extremes (`-128`/`127`) on both the max and sum paths (the sum path
specifically proving the full accumulator width is used with no premature
rounding), opcode-alternating mutual-exclusion, and randomized
backpressure independently on all three links plus a dedicated zero-stall
full-throughput case. Golden values are computed by an independently
re-derived VHDL function in the testbench, cross-checked against
`cnn_accel_model.py`'s `pool_max()`/`_pool_windows()` during authoring.

Verified standalone via GHDL (VHDL-2008; `vunit-mcp`'s shared-library
compile was blocked by an in-progress, out-of-scope sibling module —
see `cnn_accel_pool_proposal.md` "Verification backend note"): all five
test cases (`test_pool_max_kernel_sizes`, `test_pool_avg_exact_sum`,
`test_opcode_mutual_exclusion`, `test_backpressure`,
`test_full_throughput`) passed at seeds `1234` and `9999`, the last with
`stall_probability_percent_in/max/avgsum` forced to `0`. Re-run through
`vunit-mcp` once the sibling module compiles again.
