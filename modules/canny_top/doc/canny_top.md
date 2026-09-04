# canny_top

Structural top-level for the streaming Canny edge detector. Wires the full
pipeline described in `doc/canny_arch.md` behind a single flat AXI4-Stream
slave/master pair. Contains no functional logic of its own — see
`modules/canny_top/doc/canny_top_req.md` for the requirement and
`modules/canny_top/doc/canny_top_proposal.md` for the implementation
rationale (including deviations discovered/resolved during wiring: flat
ports throughout rather than records, no `g_img_width`/`g_img_height` on
`canny_sobel3x3`/`canny_nms`, and the direction-fork FIFO's
power-of-two-depth requirement).

## Generics
| Generic | Type | Purpose |
|---|---|---|
| `g_img_width` | positive | frame width in pixels |
| `g_img_height` | positive | frame height in pixels |
| `g_thresh_low` | natural | hysteresis low threshold (magnitude units) |
| `g_thresh_high` | natural | hysteresis high threshold (magnitude units) |

## Ports
| Port | Dir | Type | Purpose |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `rst_n` | in | std_logic | synchronous active-low reset |
| `s_axis_tvalid` | in | std_logic | input pixel valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(7 downto 0) | 8-bit grayscale pixel, raster order |
| `s_axis_tlast` | in | std_logic | end-of-line |
| `s_axis_tuser` | in | std_logic_vector(0 downto 0) | bit 0 = start-of-frame |
| `m_axis_tvalid` | out | std_logic | output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(7 downto 0) | bit 0 = edge ('1'/'0'), bits 7:1 = '0' |
| `m_axis_tlast` | out | std_logic | end-of-line, re-derived through the pipeline |
| `m_axis_tuser` | out | std_logic_vector(0 downto 0) | bit 0 = start-of-frame, re-derived |

## Submodule instances
`w1_inst`/`w2_inst`/`w3_inst`/`w4_inst` (`canny_window3x3`), `gaussian_inst`
(`canny_gaussian3x3`), `sobel_inst` (`canny_sobel3x3`), `dir_fifo_inst`
(hdl-modules `axi_stream.axi_stream_fifo`, unmodified), `join_inst`
(`axi_stream_join`), `nms_inst` (`canny_nms`), `threshold_inst`
(`canny_threshold`), `hysteresis_inst` (`canny_hysteresis`). See
`modules/canny_top/src/canny_top.vhd` for exact port maps.

## Status
Analyzes, elaborates, and runs cleanly under GHDL (VHDL-2008); confirmed
as part of the full-repo `run.py --compile`. `modules/canny_top/test/`
has a dedicated IP-level integration testbench (`tb_canny_top.vhd`,
`module_canny_top.py`) driven by the Python golden model
(`canny_model.py`) via `pre_config`/`post_check`, run under both
`zero_stall` and `random_stall` AXI4-Stream backpressure configs.
Confirmed genuinely GREEN (not a vacuous pass — `expected.csv`/
`result.csv` content independently inspected) as part of the full-repo
33/33 `run.py` pass.
