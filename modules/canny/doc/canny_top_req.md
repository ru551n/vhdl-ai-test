# canny_top — requirement

## Responsibility
Structural top: flat AXI4-Stream ports at the IP boundary (per
`shared/InterfaceRecords.md`), wiring the 4x `canny_window3x3` instances,
`canny_gaussian3x3`, `canny_sobel3x3`, `axi_stream_fifo` (direction-fork
elasticity), `axi_stream_join`, `canny_nms`, `canny_threshold`, and
`canny_hysteresis` into a single raster-scan pipeline with full per-link
backpressure. No behavior of its own beyond structural wiring and the
flat-to-record pack/unpack at the boundary. See `doc/canny_arch.md` for
the authoritative block diagram, inter-module interface table, and
rationale — this file exists to give `canny_top` a `modules/canny/doc/`
requirement file consistent with every other module's layout, and
duplicates only the top-level generic/port tables (kept byte-for-byte
consistent with the arch doc; the arch doc is authoritative on conflict).

## Generics
| Generic | Type | Purpose |
|---|---|---|
| `img_width` | positive | frame width in pixels |
| `img_height` | positive | frame height in pixels |
| `thresh_low` | natural | hysteresis low threshold (magnitude units) |
| `thresh_high` | natural | hysteresis high threshold (magnitude units) |

## Ports
| Port | Dir | Type | Purpose |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `reset` | in | std_logic | synchronous active-high reset |
| `s_axis_tvalid` | in | std_logic | input pixel valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(7 downto 0) | 8-bit grayscale pixel, raster order |
| `s_axis_tlast` | in | std_logic | end-of-line |
| `s_axis_tuser` | in | std_logic_vector(0 downto 0) | bit 0 = start-of-frame |
| `m_axis_tvalid` | out | std_logic | output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(7 downto 0) | bit 0 = edge ('1'/'0'), bits 7:1 = '0' |
| `m_axis_tlast` | out | std_logic | end-of-line, re-derived through the pipeline's own row/col counters |
| `m_axis_tuser` | out | std_logic_vector(0 downto 0) | bit 0 = start-of-frame, re-derived the same way |

## Protocols
Flat AXI4-Stream at the boundary; internally every link is an
`axi_stream_m2s_t`/`axi_stream_s2m_t` record pair (hdl-modules
`axi_stream_pkg`, reused directly) with `user_width=2` (SOF+border) except
the two links touching the top-level boundary itself (`user_width=1`,
matching the flat ports above) — see `doc/canny_arch.md` "Border signal
carrying (addendum, rev 2.1)". A thin pack/unpack wrapper at this entity's
boundary converts the flat ports to/from the internal record type; no
other module in this pipeline uses flat ports.

## Clock/reset
Single clock domain (`clk`), single synchronous active-high reset (`reset`)
fed unchanged to every submodule instance. No CDC in this IP; the
`axi_stream_fifo` instance is instantiated with `asynchronous => false`.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

Wires the submodules exactly per `doc/canny_arch.md`'s block diagram and
inter-module interface table:

```
s_axis -> W1 -> gaussian -> W2 -> sobel -+-> W3 -> join(a) -> nms -> threshold -> W4 -> hysteresis -> m_axis
                                          +-> axi_stream_fifo -> join(b)
```

- `W1..W4` are 4 instances of `canny_window3x3`, generic-mapped with the
  `data_width` appropriate to their position (8, 8, 11, 2) and
  `user_width` = 1 for `W1` only (fed directly from `s_axis_tuser`, 1
  bit wide), 2 for `W2..W4` (fed from an internal link that already
  carries SOF+border).
- `sobel`'s two forks (`m_axis_mag_*`, `m_axis_dir_*`) feed `W3` and
  `axi_stream_fifo` respectively; `axi_stream_fifo` is instantiated
  unmodified from hdl-modules with `asynchronous => false` and a depth
  sized to absorb the magnitude fork's extra window-fill latency (`W3`'s
  `2*img_width+2` cycles) without ever stalling the `sobel` output —
  the exact depth is a `vhfill`-time sizing decision, not fixed here.
- `axi_stream_join`'s two inputs are `W3`'s output (`data_width_a=99`)
  and `axi_stream_fifo`'s output (`data_width_b=2`); its single output
  feeds `nms`.
- `hysteresis`'s output connects directly to `m_axis_*` — no further
  wrapping needed, since `canny_hysteresis` already produces the
  top-level's exact 8-bit/1-bit-tuser format (see
  `modules/canny/doc/canny_hysteresis_req.md`).
- The flat-to-record pack/unpack at `s_axis`/`m_axis` is the only place in
  this IP that touches a flat `std_logic_vector` port for a streaming
  link; every internal link uses the record pair directly, so no
  intermediate flattening/unflattening occurs between submodule instances.
