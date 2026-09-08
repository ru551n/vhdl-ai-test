# cnn_accel_bias_requant

## Purpose

Shared output-quantization stage used after `cnn_accel_pe_array`
(`CONV2D`/`DWCONV2D`/`FC`) and after `cnn_accel_pool`'s avg-sum path
(`bias_en=0`): per lane, `int32 accumulator -> (+ bias) -> (x
requant_scale) -> (>> requant_shift, rounded) -> (+ output_offset) ->
clamp(lo, hi)`, where `(lo, hi)` is the general `[clamp_min, clamp_max]`
when `cfg_clamp_en='1'` (ISA v1.1, HW milestone H1) or the legacy
`(0 if relu_en else -128, 127)` (saturate to int8 with optional ReLU at
0). The `(requant_scale, requant_shift)` pair is per lane since ISA v1.2
(HW milestone H2): with `cfg_per_channel_en='1'` lane `l` takes its own
(multiplier, shift) from `scale_rd_data` -- the row of
`cnn_accel_weight_buffer`'s per-channel scale region addressed by this
module's own `bias_rd_addr` -- otherwise the descriptor's
`cfg_requant_scale`/`cfg_requant_shift` are broadcast to every lane (the
pre-H2 datapath bit for bit). When `requant_scale`/shift are bypassed
(`cfg_requant_en='0'`) the result is still saturated to int8, not wrapped
(architectural decision D2: a single overflow semantic across both
paths). Generic wrapper composing `math.saturate_signed` (genuinely
reused, two instances per lane -- one per path) around a bias-adder, a
requant-multiplier, and a hand-written round-half-up variable-shift
function (see "Implementation notes" for why `math.truncate_round_signed`
is not instantiated). See `cnn_accel_bias_requant_proposal.md` for the full
design rationale.

## Entity and architecture

Entity: `cnn_accel_bias_requant`. Architecture: `a`.

## Generics

| Generic | Type | Default | Meaning |
|---|---|---|---|
| `g_accum_width` | positive | 32 | Input accumulator width (int32 default). |
| `g_pe_rows` | positive | 8 | Number of parallel lanes (one per output-channel PE row). |
| `g_bias_addr_width` | positive | 1 | Width of `bias_rd_addr`. Added by vhdesign so this port can be sized to match `cnn_accel_weight_buffer`'s actual `bias_rd_addr` width at integration; the driven value is always all-zeros in v1 regardless of this generic (see "Functional behavior"). |
| `g_max_requant_shift` | natural | 31 | Upper bound on the runtime `cfg_requant_shift` value supported at full precision; larger values are clamped (defensive, not expected in compiled programs). |

Constraints (enforced by `assert ... severity failure` at elaboration):
`g_accum_width*g_pe_rows <= axi_stream_pkg.axi_stream_data_sz` (128),
`8*g_pe_rows <= axi_stream_data_sz`, `g_accum_width >= 8`,
`(15 + g_max_requant_shift) <= (g_accum_width + 1 + 32) - 2`.

## Ports

| Port | Dir | Type | Description |
|---|---|---|---|
| `clk` | in | `std_ulogic` | Single clock domain. |
| `reset` | in | `std_ulogic` | Synchronous active-high; `= reset_internal` at the top level. |
| `cfg_bias_en` | in | `std_ulogic` | From `layer_desc.flags`; sampled combinationally at beat-accept time. |
| `cfg_requant_en` | in | `std_ulogic` | Same. `'0'` selects the bypass path. |
| `cfg_relu_en` | in | `std_ulogic` | Same. Ignored while `cfg_clamp_en='1'`. |
| `cfg_clamp_en` | in | `std_ulogic` | Same (`flags` bit4, ISA v1.1/H1). `'1'` selects the general `[cfg_clamp_min, cfg_clamp_max]` clamp instead of ReLU + int8 saturate. |
| `cfg_output_offset` | in | `std_ulogic_vector(15 downto 0)` | Signed int16 (`layer_desc.output_offset`, W13[15:0], ISA v1.1/H1), added to the rounded requant result (or to the bypass sum) before the clamp. `0` = v1.0 behaviour. |
| `cfg_clamp_min` | in | `std_ulogic_vector(7 downto 0)` | Signed int8 lower clamp bound (`layer_desc.clamp_min`, W13[23:16]); used only when `cfg_clamp_en='1'`. |
| `cfg_clamp_max` | in | `std_ulogic_vector(7 downto 0)` | Signed int8 upper clamp bound (`layer_desc.clamp_max`, W13[31:24]); used only when `cfg_clamp_en='1'`. |
| `cfg_requant_scale` | in | `std_ulogic_vector(31 downto 0)` | Signed Q15 fixed-point multiplier, broadcast to every lane while `cfg_per_channel_en='0'`; ignored otherwise. |
| `cfg_requant_shift` | in | `std_ulogic_vector(7 downto 0)` | Runtime arithmetic-right-shift amount, folded with the fixed Q15 shift (15) into one combined rounding step; broadcast/ignored like `cfg_requant_scale`. |
| `cfg_per_channel_en` | in | `std_ulogic := '0'` | `flags` bit5 `PER_CHANNEL_EN` (ISA v1.2/H2). `'1'` selects the lane-wise (multiplier, shift) from `scale_rd_data` instead of the two `cfg_requant_*` ports. Sampled per beat like every other `cfg_*` port. |
| `scale_rd_data` | in | `std_ulogic_vector(c_scale_entry_width*g_pe_rows - 1 downto 0) := (others => '0')` | From `cnn_accel_weight_buffer`'s scale region, the row at `bias_rd_addr` (same tiling and 1-cycle timing as `bias_rd_data`); per lane `c_scale_entry_width` (40) bits = int32 multiplier `[31:0]` then uint8 shift `[39:32]` (`cnn_accel_pkg`). Only read while `cfg_per_channel_en='1'`; may be stale or unconnected otherwise. |
| `bias_rd_addr` | out | `std_ulogic_vector(g_bias_addr_width - 1 downto 0)` | Always all-zeros (v1 single-bias-row design decision). Also addresses the scale region. |
| `bias_rd_data` | in | `std_ulogic_vector(g_accum_width*g_pe_rows - 1 downto 0)` | From `cnn_accel_weight_buffer`; one `g_accum_width`-bit signed bias per lane, lane 0 = low bits. |
| `s_accum_m2s` | in | `axi_stream_pkg.axi_stream_m2s_t` | Accumulator input stream; low `g_accum_width*g_pe_rows` bits of `data` used. |
| `s_accum_s2m` | out | `axi_stream_pkg.axi_stream_s2m_t` | |
| `m_out_m2s` | out | `axi_stream_pkg.axi_stream_m2s_t` | int8 result stream; low `8*g_pe_rows` bits of `data` used, rest driven `'0'`; `user` driven `(others => '0')`. |
| `m_out_s2m` | in | `axi_stream_pkg.axi_stream_s2m_t` | |

## Clocking and reset

Single clock domain (`clk`). One synchronous active-high `reset` input,
clearing only the output-register's `valid` bit (`out_valid_q`); the
module has no other stateful element (the compute datapath is fully
combinational).

## Interfaces/protocols

Two independent AXI4-Stream links (`s_accum` in, `m_out` out), full
backpressure per `shared/Axi4.md`. `bias_rd_addr`/`bias_rd_data` is a
simple, non-handshaked read-address/read-data pair toward
`cnn_accel_weight_buffer` (not part of an AXI4-Stream link).

## Functional behavior

Per lane `l` (0 to `g_pe_rows - 1`), per accepted `s_accum` beat:

0. `(scale_l, shift_l) = (scale_rd_data lane l's multiplier, shift)` when
   `cfg_per_channel_en='1'` (ISA v1.2, H2), else `(cfg_requant_scale,
   cfg_requant_shift)`. The `g_max_requant_shift` clamp and the `+15` Q15
   fold below are applied to whichever was selected, so both modes see
   exactly the same shift arithmetic.
1. `total_l = accum_l + (bias_l when cfg_bias_en='1' else 0)`.
2. If `cfg_requant_en='1'`: `product_l = total_l * scale_l`
   (exact width, no truncation); `scaled_l = round_half_up(product_l >>
   (15 + clamp(shift_l, g_max_requant_shift)))` = `floor((product_l
   + 2^(S-1)) / 2^S)` with `S` the combined shift — ties round towards
   +infinity, identical to TOSA `apply_scale_32` SINGLE_ROUND (HW
   milestone H0, 2026-09-07; previously round-to-even);
   `biased_l = scaled_l + cfg_output_offset` (exact, full width; ISA v1.1,
   H1); `result_l = clamp(biased_l, lo, hi)`.
3. If `cfg_requant_en='0'` (bypass): `biased_l = total_l +
   cfg_output_offset`; `result_l = clamp(biased_l, lo, hi)` (no scaling,
   but CLAMPED like the requant path -- a single overflow semantic across
   both paths, per architectural decision D2, matching
   `cnn_accel_model.py`'s golden reference).

The clamp bounds `(lo, hi)` are `(cfg_clamp_min, cfg_clamp_max)` when
`cfg_clamp_en='1'` (`cfg_relu_en` is then ignored), else the ISA v1.0
legacy `(0 if cfg_relu_en='1' else -128, 127)` -- i.e. "ReLU before
saturate" is exactly `clamp(x, 0, 127)`, so v1.0 programs
(`cfg_output_offset=0`, `cfg_clamp_en='0'`) are bit-identical to the
pre-H1 module. The clamp is `min(max(x, lo), hi)`; the encoder rejects
`clamp_min > clamp_max`, and the RTL pins that case to `hi` like the
model does.

`bias_rd_addr` is driven constant all-zeros (see "Implementation notes").

<!-- functional-spec: hand-owned below this line -->

## Functional Description

Per lane, per accepted `s_accum` beat: `sum <= accum + (bias when
cfg_bias_en else 0)`; `scaled <= sum * cfg_requant_scale` (signed,
Q15 multiplier, so the raw product is right-shifted by 15 internally
before `cfg_requant_shift` is applied, or folded into one combined shift
amount at `vhdesign` time); `truncate_round_signed` rounds the shifted
result; `cfg_output_offset` is added to the rounded result (ISA v1.1,
H1: TOSA `output_zp`); then either the general `[cfg_clamp_min,
cfg_clamp_max]` clamp (`cfg_clamp_en='1'`) or, legacy, `saturate_signed`
clamps to the int8 range and, when `cfg_relu_en`, negative results are
clamped to 0 *before* the int8 saturate (so a large positive value still
saturates at +127, not at ReLU's unbounded upper range). When
`cfg_requant_en='0'`, the pipeline still applies bias/offset and the same
clamp (no scaling) (debug/bypass path, not
expected in normal compiled programs) -- saturating rather than wrapping
so both paths share a single overflow semantic (architectural decision
D2), matching the golden model.

## Timing/latency

**Seven cycles latency** (`c_stages = 7`): an `s_accum` beat accepted at
cycle `N` appears on `m_out` at cycle `N+7`, or whenever it next drains
against `m_out_s2m.ready`. Full throughput at zero stall on both links:
one output beat per accepted input beat, no bubbles. All seven stages
share a single `pipe_en <= (not valid_q(c_stages)) or m_out_s2m.ready`,
which also drives `s_accum_s2m.ready` — when the output stage is stalled
the whole pipeline freezes together, so nothing is dropped and no bubble
is inserted.

The stage split (S7 timing work, 2026-09-07 — previously this was a
single-cycle datapath, and its ~21 ns combinational cone was the 46.77 MHz
critical path of `cnn_accel_conv_core`):

| Stage | Work |
|---|---|
| 1 | capture the accepted beat, its bias word, its `cfg_*` values and the per-lane (multiplier, shift) selected from `scale_rd_data` vs. `cfg_requant_*` (H2) |
| 2 | bias add (`total = accum + bias`) |
| 3 | requant multiply (DSP48E1 MREG) + the bypass sum saturated to 17 bits |
| 4 | product pipeline register (bare DSP48E1 PREG, no logic); bypass: `+ cfg_output_offset` (18 bits) |
| 5 | quotient and the round-up decision (guard bit); in parallel `offset + round_up` (17 bits); bypass: saturate to int8 |
| 6 | `quotient + (offset + round_up)`, alone so its carry chain gets a full cycle (same cost as the pre-H1 incrementer) |
| 7 | saturate to int8, `clamp(lo, hi)` (general or legacy ReLU/saturate bounds), path mux, output register |

H1 (ISA v1.1) added the offset and the general clamp without a new stage:
the 16-bit offset is folded into the stage-6 rounding adder
(`round + offset == round_up + offset` pre-added at stage 5, exact on the
full-width quotient), and the clamp reuses the stage-7 saturate cone with
muxed bounds. Latency stays seven cycles.
H2 (ISA v1.2) made the multiplier and shift per lane without touching the
stage structure either: the per-channel/scalar select is a stage-1 mux
ahead of the existing capture registers, so only the stage-3 multiplier
operand and the stage-5 shift amount became lane-indexed (`scale_p`/
`shift_p` are now arrays of `g_pe_rows` lanes). Depth, throughput and
handshake are unchanged from v1.1.
Per-beat `cfg_*` values are captured *with* the beat at stage 1 rather
than read live at the stage that consumes them, because the module is
seven cycles deep: `cfg_*` may therefore change as soon as a beat has
been accepted. Throughput is unchanged by the added depth and latency
costs once per pipeline fill, not once per pixel, so the S5 frame-budget
model is unaffected.

> Note: this module's own out-of-context netlist Fmax is **not** a usable
> number — its whole datapath cone starts at input ports, which
> synthesis-only register-to-register timing never sees. It reported
> 520 MHz standalone while being `conv_core`'s 46.77 MHz critical path.
> See the `vivado-gotchas` skill.

## Registers/configuration

None (no CSR/register-file interface on this module; `cfg_*` are plain
combinational control ports, held stable for a layer's duration by the
driving `cnn_accel_layer_ctrl`, not yet designed).

## Dependencies

- `ieee.std_logic_1164`, `ieee.numeric_std`.
- `axi_stream.axi_stream_pkg` (`hdl-modules/modules/axi_stream/src/axi_stream_pkg.vhd`,
  read-only reuse) for `axi_stream_m2s_t`/`s2m_t` and `axi_stream_data_sz`.
- `math.saturate_signed` (`hdl-modules/modules/math/src/saturate_signed.vhd`,
  read-only reuse), two instances per lane (requant path and bypass path,
  both saturate to int8 -- architectural decision D2).
- Source: `modules/cnn_accel/src/cnn_accel_bias_requant.vhd`.
- Testbench: `modules/cnn_accel/test/tb_cnn_accel_bias_requant.vhd`.

## Implementation notes

- **`bias_rd_addr` is a constant all-zeros output** (v1 design decision):
  the requirement gives no tile-index/layer-start input to this module, so
  v1 assumes `out_channels <= g_pe_rows` (a single bias row covers the
  whole layer). See proposal doc §3 for the full rationale and the
  documented limitation (no multi-tile bias support in v1).
- **`math.truncate_round_signed` is not instantiated**, despite
  `doc/cnn_accel_arch.md`'s submodule table describing this module as
  composing it: its removed-LSB count is fixed by generics at elaboration,
  but the combined shift amount (`15 + cfg_requant_shift`) is a genuine
  runtime value (and, since H0, the rule is round-half-up, not
  `truncate_round_signed`'s round-to-even). The rounding is written
  inline instead.
  `math.saturate_signed` *is* genuinely reused (fixed widths regardless
  of `cfg_requant_shift`'s value). Full three-options-considered
  rationale in proposal doc §4.
- **Rounding is the guard bit, not re-multiply-and-compare** (S7 timing
  work + H0, 2026-09-07). The original `round_shift_right` recovered the
  remainder by shifting the quotient back left, subtracting, and
  comparing against half — a second variable shift, a 65-bit subtract and
  two 66-bit comparators in series, which was most of the module's 25
  CARRY4 and ~1700 LUT. S7 replaced it with the classic guard/sticky
  round-to-even form; H0 (round-half-up, ties towards +infinity, so the
  TOSA compiler can be bit-exact) then dropped the sticky reduction and
  the quotient-parity term entirely: `round_up = guard = product(shift-1)`,
  i.e. `floor((product + 2^(shift-1)) / 2^shift)`. The round-up decision
  (stage 5) stays separated from the incrementer it drives (stage 6) so
  the carry chain gets a full cycle.
  Identical results to the old function — the testbench cross-checks
  against an independently transliterated reference, not against this
  RTL's structure.
- `axi_stream_pkg`'s `data` field is a fixed 128-bit vector; this module
  asserts `g_accum_width*g_pe_rows <= 128` at elaboration. At
  `doc/cnn_accel_arch.md`'s own top-level defaults
  (`g_accum_width=32, g_pe_rows=8` => 256 bits) this does not hold --
  flagged as an open `cnn_accel_top` integration item in proposal doc §7,
  not resolved here.

## Verification notes

See `cnn_accel_bias_requant_proposal.md` §8-§9 for the full corner-case
list and verification plan, and the project's final report for the exact
`tb_cnn_accel_bias_requant` test case names and the literal `vunit-mcp`
pass/fail result. Key corner cases: round-half-up ties (both quotient
parities, both signs; `test_round_half_up_ties`), saturation both directions, ReLU-before-saturate ordering,
all 8 `bias_en`/`requant_en`/`relu_en` combinations, the
`cfg_requant_en='0'` bypass path's int8 saturation (both directions, plus
ReLU-before-saturate ordering), full-throughput/randomized-
backpressure handshake behavior, and the ISA v1.1 (H1) epilogue:
`test_output_offset_after_shift` (offset is exact and post-rounding,
saturates both ways, bypass path, random int16 offsets),
`test_general_clamp` (CLAMP_EN replaces ReLU/saturate, `relu_en` ignored,
offset + clamp together, `min = max`, full-range bounds == saturate,
random bounds, `min > max` pinned to `hi`) and
`test_clamp_en_zero_is_legacy` (garbage `clamp_min/max` with
`cfg_clamp_en='0'` has no effect), and the ISA v1.2 (H2) per-channel
select: `test_per_channel_lanes` (same accumulator on every lane with
lane-distinct multipliers 1.0/0.5/1.0>>1/-1.0 -> `x, x/2, x/2, -x`, so a
lane mix-up or a broadcast of the scalar cfg is visible immediately; then
randomized per-lane pairs incl. round-half-up ties per lane) and
`test_per_channel_en_zero_is_legacy` (a fully populated garbage table row
with `cfg_per_channel_en='0'` has no effect for every flag combination).
`tb_cnn_accel_conv_core`'s `conv3x3_offset_clamp` vector case (offset -7,
clamp [-100, 90], both bounds hit) and `conv3x3_per_channel` (six
distinct per-lane pairs from `scale_table_packed.txt`, one negative, with
the descriptor's own `requant_scale/shift` set to values that would be
wrong on any lane) cover the descriptor and scale-region plumbing end to
end. Expected
values are computed by a testbench-local reference function independently
transliterated from `cnn_accel_model.py` (not copied from this module's
own RTL structure).
