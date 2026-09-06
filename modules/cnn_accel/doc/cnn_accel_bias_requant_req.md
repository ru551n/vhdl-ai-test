# cnn_accel_bias_requant — requirement

## Responsibility

Shared output-quantization stage: `int32 accumulator -> (+ bias) -> (x
requant_scale) -> (>> requant_shift, arithmetic) -> saturate to int8 ->
(optional ReLU clamp at 0)`. Used after `cnn_accel_pe_array` for
`CONV2D`/`DWCONV2D`/`FC`, and after `cnn_accel_pool`'s sum path for
`POOL_AVG` (with `bias_en=0`). Generic wrapper composing `hdl-modules`
`math.saturate_signed` and `math.truncate_round_signed` (reused for the
shift/round and saturate steps) around a new bias-adder and
requant-multiplier.

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_accum_width` | positive | input accumulator width (int32 default) |
| `g_pe_rows` | positive | number of parallel lanes (one per output-channel PE row); `cnn_accel_pool`'s single-lane use sets this port's unused lanes idle |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `cfg_bias_en`, `cfg_requant_en`, `cfg_relu_en` | in | `std_ulogic` | from `layer_desc.flags`, latched at layer start |
| `cfg_requant_scale` | in | `std_ulogic_vector(31 downto 0)` | signed Q15, from `layer_desc.requant_scale` |
| `cfg_requant_shift` | in | `std_ulogic_vector(7 downto 0)` | from `layer_desc.requant_shift` |
| `bias_rd_addr` | out | `std_ulogic_vector` | to `cnn_accel_weight_buffer` (unused when `cfg_bias_en='0'`) |
| `bias_rd_data` | in | `std_ulogic_vector` | from `cnn_accel_weight_buffer` |
| `s_accum_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, `data` width = `g_accum_width*g_pe_rows` | opcode-muxed input: `cnn_accel_pe_array`'s accumulator or `cnn_accel_pool`'s avg-sum |
| `m_out_m2s`/`s2m` | out/in | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, `data` width = `8*g_pe_rows` | int8 result(s), to the final output `handshake_mux` (converges with `cnn_accel_pool`'s max-path) |

## Protocols

AXI4-Stream in and out, full backpressure per `shared/Axi4.md`'s
mandatory-default rule.

## Clock/reset

Synchronous active-high `reset_internal`, same abort rationale as the
other datapath modules.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

Per lane, per accepted `s_accum` beat: `sum <= accum + (bias when
cfg_bias_en else 0)`; `scaled <= sum * cfg_requant_scale` (signed,
Q15 multiplier, so the raw product is right-shifted by 15 internally
before `cfg_requant_shift` is applied, or folded into one combined shift
amount at `vhdesign` time); `truncate_round_signed` rounds the shifted
result; `saturate_signed` clamps to the int8 range; when `cfg_relu_en`,
negative results are clamped to 0 *before* the int8 saturate (so a
large positive value still saturates at +127, not at ReLU's unbounded
upper range). When `cfg_requant_en='0'`, the pipeline still applies
bias/ReLU (no scaling) and then saturates the result to int8 (debug/bypass
path, not expected in normal compiled programs) -- saturating rather than
wrapping so there is a single overflow semantic across both paths,
matching the golden model (architectural decision D2).
