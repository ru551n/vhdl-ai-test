# canny_top — proposal

## Requirements summary

Structural top only, per `modules/canny_top/doc/canny_top_req.md` and
`doc/canny_arch.md`: flat AXI4-Stream boundary, wires the 7 already-GREEN
pipeline modules (`canny_window3x3` x4, `canny_gaussian3x3`,
`canny_sobel3x3`, `canny_nms`, `canny_threshold`, `canny_hysteresis`) plus
`axi_stream_join` and hdl-modules' `axi_stream_fifo` into the single
raster-scan chain from the block diagram. No functional logic of its own.

## Deviations from `doc/canny_arch.md`'s original framing (found and resolved at wiring time)

The arch doc's "Top-level ports" section originally described every
*internal* link as an `axi_stream_m2s_t`/`axi_stream_s2m_t` record pair,
with only the top-level boundary flattened. In practice, **every one of
this project's own new modules** (`canny_window3x3`, `canny_gaussian3x3`,
`canny_sobel3x3`, `canny_nms`, `canny_threshold`, `canny_hysteresis`,
`axi_stream_join`) was implemented with flat `std_logic_vector`
`s_axis_t*`/`m_axis_t*` ports throughout, matching each module's own
req/proposal docs (written before this wiring step) rather than the
record-everywhere framing. `canny_top` follows the RTL that actually
exists: every internal link between these modules is flat, and the
record-pair convention is used **only** where unavoidable — around the
single reused-unmodified `hdl-modules axi_stream_fifo` instance (direction
fork elasticity), which has no flat-port variant. A thin
pack/unpack (`dir_fifo_pack` process + 4 concurrent unpack assignments)
brackets that one instance.

Separately: the arch doc's "Generic propagation map" states
`g_img_width`/`g_img_height` propagate into `canny_sobel3x3` and
`canny_nms` "for border bookkeeping". Neither module's actual entity (as
implemented by `vhfill`) has these generics — both instead simply pass
`tuser(1)` (border) through unchanged from their upstream
`canny_window3x3`, needing no row/column counter of their own. `canny_top`
does not pass these generics to `sobel_inst`/`nms_inst` since the ports
don't exist; only `canny_window3x3` (x4) and `canny_threshold` take
generics.

## Interface

Generics/ports: byte-for-byte identical to
`modules/canny_top/doc/canny_top_req.md` (which is itself kept consistent
with `doc/canny_arch.md`'s top-level generic/port tables).

## Architecture and dataflow

```
s_axis -> W1 -> gaussian -> W2 -> sobel -+-> W3 -> join(a) -> nms -> threshold -> W4 -> hysteresis -> m_axis
                                          +-> axi_stream_fifo -> join(b)
```

- `W1`: `canny_window3x3` (`g_data_width=>8, g_user_width=>1`), fed
  directly from `s_axis_*` (1-bit `tuser`, no border bit yet).
- `gaussian`: `canny_gaussian3x3`, no generics.
- `W2`: `canny_window3x3` (`g_data_width=>8, g_user_width=>2`).
- `sobel`: `canny_sobel3x3`, forks to `m_axis_mag_*` (11-bit, ->`W3`) and
  `m_axis_dir_*` (2-bit, -> direction FIFO).
- `W3`: `canny_window3x3` (`g_data_width=>11, g_user_width=>2`) ->
  `join`'s `s_axis_a` (99-bit).
- Direction fork: `axi_stream.axi_stream_fifo` (hdl-modules, unmodified,
  `asynchronous=>false`), record-packed/unpacked around the flat
  `sobel_dir_*`/`fifo_dir_*` signals. `data_width=>2, user_width=>2`.
  Depth: `next_pow2(2*g_img_width + 16)` — hdl-modules' underlying
  `fifo.vhd` asserts "RAM depth must be a power of two" (discovered this
  session at elaboration time, not documented in the req/arch docs), so a
  local `next_pow2` function rounds the arch doc's "absorb W3's
  `2*g_img_width+2` cycles" sizing guidance up to a legal depth.
- `join`: `axi_stream_join` (`g_data_width_a=>99, g_data_width_b=>2`) ->
  `nms`.
- `nms`: `canny_nms`, no generics.
- `threshold`: `canny_threshold` (`g_thresh_low`, `g_thresh_high`
  propagated from `canny_top`'s own generics).
- `W4`: `canny_window3x3` (`g_data_width=>2, g_user_width=>2`) ->
  `hysteresis`.
- `hysteresis`: `canny_hysteresis` -> `m_axis_*` directly (already
  produces the top-level's exact 8-bit-data/1-bit-`tuser` format).

## Clock/reset behavior

Single clock domain (`clk`), single synchronous active-low `rst_n` fed
unchanged to every submodule instance (including the reused
`axi_stream_fifo`, itself stateless w.r.t. `rst_n` since it has no reset
port at all — resets to empty via its underlying `fifo.vhd`'s own internal
init, not an explicit reset input). No CDC; `asynchronous=>false`.

## Verification status

This file's RTL has been confirmed to **analyze, elaborate, and run (no
assertion failures) under GHDL** with `g_img_width=16, g_img_height=16,
g_thresh_low=50, g_thresh_high=100`, both as part of the full-repo
`run.py --compile` (all 8 modules + this one) and a standalone
`ghdl -e`/`ghdl -r --stop-time=1us` smoke check with no stimulus applied
(reset held, all inputs at their default '0'/'U').

Beyond that structural smoke check, `modules/canny_top/test/tb_canny_top.vhd`
+ `module_canny_top.py` implement a full IP-level integration test driven
by the Python golden model (`canny_model.py`'s `canny_pipeline`) via VUnit
`pre_config`/`post_check`: `pre_config` writes a random 20x16 stimulus
frame plus the model's precomputed `expected.csv`, the testbench streams
it through `canny_top` over raw `vunit_lib.axi_stream_master`/
`axi_stream_slave` VCs (1-bit `tuser`, so the hdl-modules `bfm.axi_stream_*`
wrappers weren't a fit), and `post_check` re-parses both `expected.csv`
and the DUT's `result.csv` with `csv.reader` and compares them cell by
cell. Two configs — `zero_stall` and `random_stall` (randomized
`stall_config` on both AXI4-Stream sides) — both pass as part of the
full-repo 33/33 `run.py` result. The 20x16 frame size was chosen
deliberately: the pipeline's 4 chained `canny_window3x3` stages each
dilate the forced-zero border ring by 1 pixel, so a too-small frame
(e.g. the first-draft 10x8) leaves zero interior pixels and produces a
vacuously all-zero `expected.csv` that passes "for free" without
comparing any real edge value — caught by manually inspecting
`expected.csv`'s content (30-35 nonzero pixels out of 320) rather than
just trusting the pass/fail summary, per the project's standing
"never trust a self-reported all-green without inspecting the actual
artifact" rule.
