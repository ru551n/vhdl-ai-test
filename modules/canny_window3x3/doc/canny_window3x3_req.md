# canny_window3x3 — requirement

## Responsibility
Generic reusable AXI4-Stream 3x3 sliding-window generator over a continuous
raster-scan stream of `g_img_width x g_img_height` samples, with full
elastic backpressure (`TREADY`) on both sides. Reused by four pipeline
stages (W1 raw, W2 smoothed, W3 magnitude, W4 classification) with
different `g_data_width`.

## Generics
| Generic | Type | Notes |
|---|---|---|
| `g_img_width` | positive | frame width |
| `g_img_height` | positive | frame height |
| `g_data_width` | positive | per-sample bit width |
| `g_user_width` | positive range 1 to 2 | width of `s_axis_tuser`. `1` only for the chain's first instance (fed directly from the top-level boundary, which carries SOF only); `2` for every other instance (fed from a link that already carries SOF + border). When `1`, the incoming border contribution for all 9 window taps is treated as constant `'0'` (nothing has been marked border yet upstream). `m_axis_tuser` is always 2 bits wide regardless of this generic. |

## Ports
| Port | Dir | Type |
|---|---|---|
| `clk` | in | std_logic |
| `rst_n` | in | std_logic |
| `s_axis_tvalid` | in | std_logic |
| `s_axis_tready` | out | std_logic |
| `s_axis_tdata` | in | std_logic_vector(g_data_width-1 downto 0) |
| `s_axis_tuser` | in | std_logic_vector(g_user_width-1 downto 0) — bit 0 = SOF; bit 1 (only present when `g_user_width=2`) = border |
| `s_axis_tlast` | in | std_logic — end-of-line (EOL) |
| `m_axis_tvalid` | out | std_logic |
| `m_axis_tready` | in | std_logic |
| `m_axis_tdata` | out | std_logic_vector(9*g_data_width-1 downto 0) — 3x3 window, packed row-major MSB-to-LSB: `w_tl` at the top bits ... `w_br` at the bottom `g_data_width` bits |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) — bit 0 = SOF (passthrough), bit 1 = border (freshly computed, see Functional Description) |
| `m_axis_tlast` | out | std_logic — EOL, recomputed from this module's own output-side row/column counters |

## Protocols
Full AXI4-Stream elastic handshake on both sides per `shared/Axi4.md`; no
free-running/clock-enable shortcut. Internally realized as 2 elastic
line-buffer FIFOs (reusing hdl-modules `fifo.fifo_wrapper`, the same
primitive `axi_stream_fifo` wraps) plus 2-deep column-tap registers, all
gated by `tready`/`tvalid` so a stall on `m_axis_tready` stops the line
buffers and column taps from advancing that cycle — no bubble insertion, no
re-fetch, no dropped or duplicated sample. `s_axis_tuser`/`s_axis_tlast`
ride alongside `s_axis_tdata` through the same elastic lanes (extra bit
lanes through the same FIFOs/registers) so they share identical
backpressure gating and can never desync from the data they describe.

## Clock/reset
Single clock, synchronous active-low reset clears all internal buffers,
FIFOs, and the output-side raster-position counters.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

- Internally realizes 2 row (line) buffers of depth `g_img_width` (elastic
  FIFOs, not shift registers, so they hold their contents under
  backpressure) plus a 2-deep horizontal tap register per row (top/middle/
  bottom), so the window becomes available once `2*g_img_width + 2`
  accepted (`tvalid and tready`) input beats have occurred, in the same
  raster order (1:1, no drops, no duplicates).
- `m_axis_tvalid` follows `s_axis_tvalid`'s accepted-beat image exactly
  (1:1 once the pipeline has filled), so exactly `g_img_width*g_img_height`
  output beats occur per frame.
- `m_axis_tuser(0)` (SOF) passes through unchanged, delay-matched to the
  center tap (`w_mm`) through the same elastic lanes as the data.
- `m_axis_tuser(1)` (border) = this module's own edge-of-frame test
  (evaluated on its own output-side raster counter: this output position is
  row 0, row `g_img_height-1`, col 0, or col `g_img_width-1`) **OR**'d with
  an OR-reduction across **all 9** of the incoming `s_axis_tuser(1)` bits
  that rode alongside the 9 taps being windowed (constant `'0'` for all 9
  when `g_user_width=1`). This is a proper dilate of the border ring by
  1 pixel per instance, not just a copy of the center tap's border flag —
  see `doc/canny_arch.md` "Growing border (corrected, rev 2.2)" for the
  rationale and the bug this replaces.
- `m_axis_tlast` (EOL) is recomputed from this module's own output-side
  column counter (`col = g_img_width-1`), not assumed to survive
  windowing unchanged from `s_axis_tlast`.
- `canny_window3x3` recomputes its own row/column counters from the stream
  of *accepted* `s_axis_tlast` pulses (does not trust `g_img_width` alone
  against a free-running counter that could desync under stalls) to derive
  both `border` and `tlast` status for the emitted window.
- Out-of-bounds window taps (for samples before the stream starts, or
  belonging to a row/column outside the frame) read as `(others => '0')`
  within `m_axis_tdata`; their value is not meaningful and downstream
  logic must gate on `m_axis_tuser(1)` rather than relying on tap values
  at the border.
