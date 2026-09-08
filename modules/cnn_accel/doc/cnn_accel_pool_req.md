# cnn_accel_pool — requirement

## Responsibility

Spatial reduction over a `cnn_accel_window_gen`-produced pooling window:
`POOL_MAX` computes the int8 maximum of the window's taps and emits it
directly (already a valid int8 output, no further scaling); `POOL_AVG`
computes the int32 sum of the window's taps and emits it for
`cnn_accel_bias_requant` to scale/round into int8 (division by the pool
area is exactly a requantize operation, see
`doc/cnn_accel_arch.md` "Non-obvious boundary rationale").

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_max_kernel_size` | positive | upper bound on pool `K_h*K_w`, sizes the reduction tree. This is the **pool** kernel bound (`cnn_accel_constant_max_pool_kernel_size` = 5, for YOLOv8n's 5x5 SPPF pool), deliberately separate from and larger than the convolution datapath's own bound of 3 -- see `doc/cnn_accel_top_v2_arch.md` §12a |
| `g_accum_width` | positive | `POOL_AVG` sum width |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `cfg_opcode` | in | `std_ulogic_vector(7 downto 0)` | `POOL_MAX` vs `POOL_AVG` |
| `cfg_pool_kernel_h`/`w` | in | `std_ulogic_vector(7 downto 0)` | pool area, for `POOL_AVG`'s implicit divide |
| `s_window_m2s`/`s2m` | in/out | `cnn_accel_pkg.window_m2s_t`/`window_s2m_t`, `data(0 to g_max_kernel_size**2 - 1)` | from `cnn_accel_window_gen`, one channel lane's taps. **Was** `axi_stream_m2s_t`; its fixed 128-bit payload capped the pool kernel at 3 (5x5 = 200 bits), so this port moved to the unconstrained tap-array record the conv path already uses rather than widening the hdl-modules-wide `axi_stream_data_sz` -- see `doc/cnn_accel_top_v2_arch.md` §12a |
| `m_max_m2s`/`s2m` | out/in | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, `data` width = 8 | `POOL_MAX` result, to the final output `handshake_mux` (bypasses `cnn_accel_bias_requant`) |
| `m_avgsum_m2s`/`s2m` | out/in | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, `data` width = `g_accum_width` | `POOL_AVG` sum, to `cnn_accel_bias_requant` |

## Protocols

Valid/ready window record in (same handshake rules), two
mutually-exclusive AXI4-Stream outputs (only one
active per instruction, selected by `cfg_opcode`), full backpressure per
`shared/Axi4.md`'s mandatory-default rule.

## Clock/reset

Synchronous active-high `reset_internal`, same abort rationale as the
other datapath modules.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

`POOL_MAX`: combinational/pipelined max-reduction tree over the window's
taps, registered onto `m_max_m2s`. `POOL_AVG`: adder-tree sum over the
window's taps, registered onto `m_avgsum_m2s`; `cnn_accel_layer_ctrl`
programs `cnn_accel_bias_requant`'s `requant_scale`/`requant_shift` for
that instruction so the sum is divided by `pool_kernel_h*pool_kernel_w`
(exact power-of-two areas reduce to a pure shift; non-power-of-two areas
use `requant_scale` as a fixed-point reciprocal, same mechanism as
ordinary conv/fc requantization).
