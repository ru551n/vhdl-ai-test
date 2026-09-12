# flash_model FFI contract

The single interface between the VHDL verification component
(`sim/flash_model.vhd`) and the Python device model
(`python/flash_model_bridge.py`). Both sides are generated from this document;
a change here is a change to both, guarded at run time by `layout_version()`.

`LAYOUT_VERSION = 1`

## Call shapes

The VUnit bridge allows **zero or one positional scalar argument** (or any
number of positional `integer_array_t`), one return value per call. Everything
else goes through `kw()`. Every function below therefore takes `id` via `kw`.

| function | args | returns |
|---|---|---|
| `layout_version()` | — | `int` |
| `flash_create(...)` | kw: `profile`, `size_bytes`, `page_bytes`, `sector_bytes`, `block_bytes`, `addr_bytes`, `jedec_id` | `int` instance id |
| `flash_reset(id)` | kw: `id` | `0` |
| `cs_assert(id, now_s)` | kw: `id`, `now_s` | `int` packed directive |
| `xfer(id, byte_in, now_s)` | kw: `id`, `byte_in`, optional `now_s` | `int` packed directive |
| `cs_deassert(id, trailing_bits, now_s)` | kw: `id`, `trailing_bits`, `now_s` | `real` busy seconds |
| `get_timing_limits(id)` | kw: `id` | `int32[]` picoseconds |

`byte_in` is `-1` when the VC is clocking a byte **out** rather than in.

## Packed directive (the `int` returned by `cs_assert` / `xfer`)

Little-end-first bit packing into a non-negative 32-bit integer:

| field | width | shift | values |
|---|---|---|---|
| `action` | 2 | 0 | 0 = receive, 1 = transmit, 2 = ignore_rest |
| `lanes` | 3 | 2 | 1, 2 or 4 — applies to **this** action |
| `pre_dummy_cycles` | 6 | 5 | 0..63 SCK cycles, IOs Hi-Z, **before** this action |
| `byte_out` | 8 | 11 | valid when `action = transmit` |
| `flags` | 4 | 19 | bit 0 = `volatile` (pass `now_s` on the next `xfer`) |
| `n_bytes` | 10 | 23 | reserved for chunking; always 1 in the MVP |

Every field describes the same, next action — one tense throughout.

`pre_dummy_cycles` is a **prefix on the next action**, never a phase of its own.
That is what lets `0x6B` (address x1 -> 8 dummy -> data x4) be one directive,
and it also covers `0x77`, where dummy cycles precede a *receive*.

`ignore_rest` means: drive nothing, consume clocks until CS rises.

## Timing limits (`get_timing_limits` -> int32[], picoseconds)

Fixed order; VHDL indexes these with named constants, never literals.
A value of `0` means "not specified, do not check".

| idx | name | kind |
|---|---|---|
| 0 | `t_sck_min_ps` | check (min SCK period) |
| 1 | `t_sck_high_min_ps` | check |
| 2 | `t_sck_low_min_ps` | check |
| 3 | `t_slch_ps` | check (CS low to first SCK edge) |
| 4 | `t_chsh_ps` | check (last SCK edge to CS high) |
| 5 | `t_shsl_ps` | check (CS deselect between commands) |
| 6 | `t_dvch_ps` | check (data in setup) |
| 7 | `t_chdx_ps` | check (data in hold) |
| 8 | `t_clqv_ps` | **delay** (clock to output valid) |
| 9 | `t_shqz_ps` | **delay** (CS high to output Hi-Z) |

Indices 0..7 are asserted against DUT-driven pins. Indices 8..9 are scheduled as
`after` on VC-driven pins -- asserting on them would make the VC fail on its own
output.

## Control plane

| function | args | returns |
|---|---|---|
| `preload(data, id, addr)` | positional `int32[]` data; kw `id`, `addr` | `0` |
| `preload_fill(id, addr, num_bytes, value)` | kw | `0` |
| `load_image(id, path, fmt, base)` | kw | `0` |
| `read_back(id, addr, num_bytes)` | kw | `int32[]` |
| `check_content(expected, id, addr)` | positional `int32[]`; kw `id`, `addr` | `0` |
| `check_content_fill(id, addr, num_bytes, value)` | kw | `0` |
| `written_regions(id)` | kw | `int32[]` flat `[addr, len, ...]` |
| `set_timing_enable(id, enable)` | kw | `0` |
| `set_timing(id, name, seconds)` | kw | `0` |
| `set_protection(id, addr, num_bytes, locked)` | kw | `0` |
| `get_stat(id, name)` | kw | `int` |

Every side-effecting call returns `0` because the bridge has no
"takes arguments, returns nothing" form.

## Error policy

Python **raises**; it never returns a status code. The bridge turns an uncaught
exception into a VUnit FAILURE carrying the full traceback. VHDL never wraps a
`python_call` in `check_true`. Pin-level timing violations are the exception:
those are detected in VHDL and reported through the VC's own `checker_t`, so
negative tests can `mock`/`unmock` the logger.
