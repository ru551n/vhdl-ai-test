# cnn_accel_pe_array — design proposal (`vhdesign`)

## 1. Requirements summary

int8 x int8 MAC array, `g_pe_rows x g_pe_cols` parallelism, `g_accum_width`
(int32 default) accumulation. Consumes one window beat (from
`cnn_accel_window_gen`, already opcode-routed to this module by a
`handshake_splitter`) and sequences `weight_rd_addr` over the window's taps,
grouped `g_pe_cols` at a time, accumulating into `g_pe_rows` parallel
accumulators (one per output channel currently in flight). `CONV2D`/`FC`
mode sums across all `in_channels`; `DWCONV2D` mode keeps each accumulator
scoped to a single input channel (no cross-channel sum). Once a window's
MAC sequence completes, the `g_pe_rows` accumulators are emitted as one
`m_accum_m2s` beat (`last` mirrors the window's `last`). Per
`modules/cnn_accel/doc/cnn_accel_pe_array_req.md`, exact DSP-packing/
pipelining microarchitecture is deferred to this document.

**Naming note**: per this round's updated project convention, this
module's RTL/testbench files and entities use plain names (`pe_array`/
`tb_pe_array`), not the `cnn_accel_pe_array` prefix used by the sibling
leaf modules (`cnn_accel_pool`, `cnn_accel_bias_requant`,
`cnn_accel_window_gen`, `cnn_accel_weight_buffer`) — those keep their
existing names unchanged. Only this and future new `cnn_accel` submodules
use the plain form. Requirement/proposal/final docs keep the
`cnn_accel_pe_array_*` doc filenames for continuity with the existing doc
set.

## 2. Interface (as given by the requirement, gaps resolved below)

| Generic | Type | Source |
|---|---|---|
| `g_pe_rows` | `positive` | requirement |
| `g_pe_cols` | `positive` | requirement |
| `g_accum_width` | `positive` | requirement |
| `g_max_kernel_size` | `positive` | requirement |
| `g_weight_buffer_depth` | `positive` | **added by vhdesign** — see §3.1 |

| Port | Dir | Type | Source |
|---|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | requirement |
| `cfg_opcode` | in | `std_ulogic_vector(7 downto 0)` | requirement |
| `cfg_in_channels`, `cfg_out_channels` | in | `std_ulogic_vector(15 downto 0)` | requirement |
| `cfg_kernel_h`, `cfg_kernel_w` | in | `std_ulogic_vector(7 downto 0)` | **added by vhdesign** — see §3.2 |
| `s_window_m2s`/`s2m` | in/out | `axi_stream_m2s_t`/`s2m_t` | requirement |
| `weight_rd_addr` | out | `std_ulogic_vector(num_bits_needed(g_weight_buffer_depth - 1) - 1 downto 0)` | requirement (width resolved, §3.1) |
| `weight_rd_data` | in | `std_ulogic_vector(8*g_pe_rows*g_pe_cols - 1 downto 0)` | requirement |
| `m_accum_m2s`/`s2m` | out/in | `axi_stream_m2s_t`/`s2m_t` | requirement |

## 3. Design decisions (resolving requirement ambiguities)

### 3.1 `weight_rd_addr` width: `g_weight_buffer_depth` generic added

The requirement leaves `weight_rd_addr`'s width unspecified
("`std_ulogic_vector`"). `cnn_accel_weight_buffer`'s own
`weight_rd_addr` is `num_bits_needed(g_weight_buffer_depth - 1)` bits wide
(`modules/cnn_accel/src/cnn_accel_weight_buffer.vhd:64`) — a row/tile
address into its weight region. `cnn_accel_pe_array` is the sole driver of
that port, so it needs the same generic to size its own output the same
way, exactly the precedent `cnn_accel_bias_requant` already set for
`bias_rd_addr`/`g_bias_addr_width` (its own header comment: "Added by
vhdesign ... so this module's read-address port can be sized to match
`cnn_accel_weight_buffer`'s actual ... width at `cnn_accel_top`
integration time"). `num_bits_needed` comes from `math.math_pkg` (the same
import `cnn_accel_weight_buffer.vhd` and `cnn_accel_bias_requant.vhd`
already use) — no hand-rolled width helper.

An elaboration-time assert bounds the number of weight rows this module
will ever address (`num_groups`, computed at runtime — see §5) to
`g_weight_buffer_depth`, checked (as a runtime assert, since `num_groups`
depends on runtime `cfg_*`) at the moment a window's MAC sequence starts.

### 3.2 `cfg_kernel_h`/`cfg_kernel_w` ports added

The requirement's functional description says this module "sequences
`weight_rd_addr` over the window's `K_h*K_w*in_channels` taps" but the
Ports table has no `K_h`/`K_w` input — `cfg_in_channels`/`cfg_out_channels`
alone cannot express the kernel shape. Every other module that needs the
kernel shape at runtime has its own `cfg_kernel_h`/`cfg_kernel_w`
(`cnn_accel_window_gen`) or `cfg_pool_kernel_h`/`cfg_pool_kernel_w`
(`cnn_accel_pool`) input; `cnn_accel_pe_array` needs the equivalent to
size its own MAC sequence and to bound-check the window's tap count
against the fixed 128-bit `axi_stream_pkg` data field (same contract
`cnn_accel_window_gen`/`cnn_accel_pool` already assert). Added as two
`std_ulogic_vector(7 downto 0)` inputs, matching `cnn_accel_window_gen`'s
own port widths/types for the same fields (so `cnn_accel_layer_ctrl` can
fan the same decoded `layer_desc.kernel_h`/`kernel_w` out to both modules
unchanged).

No `start`/latch-pulse port is added: like `cnn_accel_pool` (which has the
identical "cfg latched at layer start" wording in its own requirement but
no `start` port either), `cfg_*` here are plain level inputs, sampled
combinationally at `s_window` accept time and captured into internal
`_q` registers so they stay stable for the whole (multi-cycle) MAC
sequence even if the caller changes them before the next window (the
window-gen precedent's actual latching mechanism, driven by its own
`start` pulse, is a per-frame concern; this module's "latch" is per-window
and self-contained).

### 3.3 `cfg_out_channels` unused in the datapath (v1 simplification)

Kept on the port list (the requirement lists it), but this module always
computes all `g_pe_rows` accumulator lanes regardless of
`cfg_out_channels`; it does not mask/gate lanes `>= cfg_out_channels`.
This mirrors the *same* `out_channels <= g_pe_rows` single-tile
simplification `cnn_accel_bias_requant` and `cnn_accel_weight_buffer`
already made explicit for `bias_rd_addr` (always `0`, "assumes a single
bias row ... covers all `g_pe_rows` output channels for the whole layer")
— multi-tile output-channel iteration (`out_channels > g_pe_rows`) is out
of scope for v1 across this whole IP, not just this module, and is an
open item for a later round (recorded in §12).

### 3.4 Weight-lane and window-tap indexing conventions (this module's own choice)

Two things are genuinely new conventions (neither `cnn_accel_weight_buffer`
nor `cnn_accel_window_gen` constrains them further):

- **Weight lane index** within `weight_rd_data`/one fetched row: lane
  `l = r * g_pe_cols + c` (row-major: PE row `r` in `0 .. g_pe_rows-1`,
  column `c` in `0 .. g_pe_cols-1`), at bits `8*(l+1)-1 downto 8*l` — same
  "row-major, ascending from low bits" convention used everywhere else in
  this IP (`cnn_accel_window_gen`'s tap index, `cnn_accel_pool`'s tap
  index). Whoever writes the weight-loading compiler/DMA descriptor logic
  for `cnn_accel_weight_buffer`'s fill side must produce rows in this lane
  order — an integration-time contract, out of scope for this module's own
  correctness (this module only *reads* `weight_rd_data` with this
  convention).
- **MAC group sequencing** ("`weight_rd_addr` sequenced over
  `K_h*K_w*in_channels` taps grouped `g_pe_cols` at a time"): one *row*
  (weight-buffer address) supplies one group of `g_pe_cols` columns for
  *all* `g_pe_rows` output channels/lanes at once (`g_pe_rows * g_pe_cols`
  weights per row, matching `weight_rd_data`'s width exactly). Address
  `0` is the first group of the window's MAC sequence and addresses
  restart at `0` for every new window (weights are per-layer constants,
  reused unchanged across every window/output-pixel position of that
  layer — same "row 0 covers the whole layer" idea `cnn_accel_bias_requant`
  already uses for bias, generalized to `num_groups` rows instead of one).

### 3.5 `CONV2D`/`FC` vs `DWCONV2D` grouping (the requirement's own distinction, made concrete)

`cnn_accel_window_gen`'s `m_window_m2s.data` packs tap `i` (row-major
spatial index, `i = row*kernel_w + col`), channel `c`, at bit offset
`8*(i*in_channels + c)` — **this is used unmodified by both opcode
families below; `cnn_accel_window_gen` always packs the full
`kernel_h*kernel_w*in_channels` tap set regardless of downstream opcode**
(it has no opcode input at all — see `doc/cnn_accel_arch.md`'s block
diagram, the same window-gen output feeds both `cnn_accel_pe_array` and
`cnn_accel_pool`). Consequently the **same** runtime bound applies to
both modes: `kernel_h * kernel_w * in_channels <= axi_stream_data_sz / 8`
(`= 16`, the same 128-bit-beat contract `cnn_accel_window_gen`/
`cnn_accel_pool` already assert on their own generics) — checked here as
a runtime `assert` (generics alone don't bound it; `cfg_in_channels` is a
runtime value), at the moment a window is accepted.

- **`CONV2D`/`FC`** ("sums across all `in_channels`"): the flat MAC index
  visited by group `g`, column `c` is `t = g*g_pe_cols + c`, ranging over
  `0 .. kernel_h*kernel_w*in_channels - 1`. This is **exactly** the window
  packing's own flat index (`i*in_channels + c_channel`), so the operand
  for `(g, c)` is simply `window(8*t + 7 downto 8*t)` — no recombination
  needed. `num_groups = ceil(kernel_h*kernel_w*in_channels / g_pe_cols)`.
  Every lane `r` (output channel `oc_base + r`) uses the **same** operand
  (`t`) and its **own** weight (`weight_rd_data` lane `r*g_pe_cols+c`) —
  the conventional "broadcast activation, per-lane weight" systolic/
  input-stationary shape.
- **`DWCONV2D`** ("no cross-channel accumulation", one filter per
  channel, `out_channels == in_channels`): only `kernel_h*kernel_w`
  *distinct spatial taps* are summed per lane, but each lane `r` (channel
  `r`, `0 <= r < in_channels`, single-tile per §3.3) must read a
  **different** operand than every other lane — its own channel's pixel.
  Group `g`, column `c` selects spatial tap `t = g*g_pe_cols + c`
  (`0 .. kernel_h*kernel_w - 1`, so `num_groups = ceil(kernel_h*kernel_w /
  g_pe_cols)`, generally *fewer* groups than `CONV2D` for the same kernel
  since there is no `in_channels` factor); lane `r`'s operand is
  `window(8*idx + 7 downto 8*idx)` with `idx = t*in_channels + r` — i.e.
  the **same** window-packing formula as `CONV2D`, just with the channel
  index fixed per-lane (`r`) instead of swept by `c`. Because `r <
  in_channels` (asserted via the `valid` guard, not a hard failure — an
  out-of-range lane simply contributes `0`) and `t < kernel_h*kernel_w`,
  `idx <= kernel_h*kernel_w*in_channels - 1`, the same bound already
  asserted above — no separate bound needed for this mode. Per-lane weight
  is still `weight_rd_data` lane `r*g_pe_cols + c` (one weight per lane
  per group, same physical read port/row shape as `CONV2D`) — a
  `DWCONV2D` weight row only ever needs `in_channels` valid lanes'
  weights populated per group (the rest, if `g_pe_rows > in_channels`,
  are simply never read since those lanes are masked invalid); this
  module does not care what the weight buffer put in the unused lanes.

This means the PE array itself does not need two structurally different
data paths: a single per-`(r, c)` "select operand index, select validity"
function (parameterized by opcode) followed by the identical
multiply-accumulate, described precisely in §6.

### 3.6 No compute/output overlap across more than one window in flight

The MAC-sequencing FSM (one active window's worth of state: latched
`cfg`/window register, group counter, `g_pe_rows` accumulators) is a
single instance — only one window is ever "in flight" through the compute
engine. The completed-but-not-yet-drained result is held in a **separate**
one-entry output register (`out_valid_q`/`out_data_q`, same shape as
`cnn_accel_pool`'s single tagged output register, minus the tag since
there is only one output stream here) so the compute engine *can* already
start the next window's MAC sequence while the previous result waits on
`m_accum_s2m.ready` — see §5's state machine. Deeper pipelining (e.g. a
second compute-engine instance, or overlapping weight-address issue for
window `N+1` before window `N`'s last accumulate finishes) is not
implemented; `doc/cnn_accel_arch.md`'s "Open items" already defers "Exact
PE-array microarchitecture (output-stationary vs input-stationary, DSP
packing of 2 int8 MACs per DSP48)" to this document, and this is the
concrete, minimal choice made here (documented as an explicit deviation,
§12).

## 4. Clock/reset

Single clock domain (`clk`). Synchronous, active-high `reset`
(`reset_internal` at the IP top level) — required by the requirement's
own "Clock/reset" section ("partial-sum accumulator registers must not
leak a stale sum into the next layer/tile after a host abort"): `reset`
clears the FSM state to `idle`, the group counter, and `out_valid_q`
(dropping any in-flight/pending accumulation and any pending-but-undrained
output beat) — the accumulator registers' *content* does not need
explicit reset (only ever read while a state machine that *is* reset
guarantees they hold a value written during the current, post-reset MAC
sequence).

## 5. Architecture and dataflow — state machine

Three states, `state_t = (idle, run, done)`:

```text
                s_window accept
        +-----> idle -------------------+
        |        ^                      |
        |        | commit (output free) v
        |      done <------------- run (num_groups+1 cycles)
        |        |  (output busy: stay)
        +--------+  commit (output free)
```

- **`idle`**: `s_window_s2m.ready <= '1'` unconditionally (the compute
  engine and the one-entry output register are decoupled — see §3.6, so
  idle can accept a new window even while a previous result is still
  parked in the output register awaiting `m_accum_s2m.ready`). On accept
  (`s_window_m2s.valid = '1'` this cycle): latch `cfg_opcode`,
  `cfg_in_channels`, `cfg_kernel_h`, `cfg_kernel_w`, the window data, and
  `last`, into `_q` registers; compute (one-shot, plain integer
  arithmetic, exactly like `cnn_accel_window_gen`'s own `out_width_q`/
  `out_height_q` one-shot computation at `start`) `total_window_taps =
  kernel_h*kernel_w*in_channels` (asserted `<= 16`), `mac_taps =
  total_window_taps` (`CONV2D`/`FC`) or `kernel_h*kernel_w` (`DWCONV2D`),
  and `num_groups = ceil(mac_taps / g_pe_cols)` (asserted
  `<= g_weight_buffer_depth`); reset all `g_pe_rows` accumulators to `0`;
  reset the group counter to `0`; move to `run`.
- **`run`**: group counter `cycle_q` runs `0 .. num_groups` inclusive
  (`num_groups + 1` cycles total). While `cycle_q <= num_groups - 1`,
  `weight_rd_addr <= cycle_q` (issuing the address for group `cycle_q`,
  sampled by `cnn_accel_weight_buffer`'s registered read this same edge —
  data available starting next cycle). Whenever `cycle_q >= 1`,
  `weight_rd_data` this cycle holds group `cycle_q - 1`'s row (issued the
  previous cycle): compute that group's per-lane partial product sum
  (§6) and add it into the accumulators. When `cycle_q = num_groups` (the
  last group's accumulate is being applied this cycle): if the output
  register can accept this cycle (`out_valid_q = '0'` or `m_accum_s2m.ready
  = '1'`), commit directly (pack the just-computed final accumulator
  values into `out_data_q`, `out_valid_q <= '1'`, `out_last_q <=
  window_last_q`) and return to `idle` next cycle; otherwise move to
  `done` (holding the final accumulator values, computed this cycle, in
  `accum_q` — no further arithmetic needed).
- **`done`**: waits for the output register to free (`out_valid_q = '0'`
  or the current cycle drains it); on that cycle, commit `accum_q` into
  `out_data_q`/`out_valid_q`/`out_last_q` as above and return to `idle`.

The output register itself follows `cnn_accel_pool`'s own one-entry
register idiom: a drain clears `out_valid_q` to `'0'` unless the same
cycle also commits a new result (in which case the register is refilled
with no bubble) — see the RTL's `register_stage`-equivalent ordering.

## 6. Algorithms — per-lane MAC (shared by both opcode modes)

For PE row `r` (`0 .. g_pe_rows-1`) and column `c` (`0 .. g_pe_cols-1`),
group `g` (`= cycle_q - 1`, the group whose weights are valid this
cycle):

```text
if is_dwconv2d:
  spatial_t := g*g_pe_cols + c
  idx       := spatial_t * in_channels + r
  valid     := (spatial_t < mac_taps) and (r < in_channels)
else:
  idx       := g*g_pe_cols + c
  valid     := idx < mac_taps

operand := signed(window_q(8*idx+7 downto 8*idx)) if valid else 0
weight  := signed(weight_rd_data(8*(r*g_pe_cols+c)+7 downto 8*(r*g_pe_cols+c)))
product := operand * weight                          -- 16-bit exact
accum_q(r) += resize(product, g_accum_width) if valid else 0
```

(the RTL clamps `idx` to `0` before slicing whenever `valid = false`, so
an out-of-range computed index — e.g. `g_pe_cols` larger than the
window's actual remaining tap count in the last group — never produces an
illegal vector slice, purely a defensive/simulation-safety measure; the
runtime asserts in §3.5/§5 keep every *valid* `idx` within
`window_q`'s 128 bits by construction).

`weight_rd_data` lane `r*g_pe_cols + c` is read unconditionally
(regardless of `valid`) — reading an unused lane is harmless, its product
is simply discarded by the `valid` gate.

## 7. Numeric types and widths

- `window_q`: `std_ulogic_vector(axi_stream_data_sz-1 downto 0)` (128
  bits, the fixed `axi_stream_pkg` field), holding at most 16 signed int8
  taps (per the §3.5 bound).
- `operand`/`weight`: `signed(7 downto 0)` (int8).
- `product`: `signed(15 downto 0)` (exact int8 x int8, no truncation).
- `accum_q(r)`: `signed(g_accum_width-1 downto 0)` (int32 by default);
  `resize(product, g_accum_width)` before adding — no intermediate
  saturation (matches `cnn_accel_pool`'s `reduce_sum` precedent: overflow
  of `g_accum_width` is a configuration error, not handled here, same as
  every other accumulate-only stage in this IP).
- `cycle_q`, `num_groups_q`, `mac_taps_q`, `total_window_taps_q`: plain
  `natural range 0 to 16` (spec'd via a `c_max_taps = axi_stream_data_sz /
  8 = 16` constant) — not `unsigned` bit vectors, matching
  `cnn_accel_pool`'s own `active_count : natural range 0 to c_max_taps`
  precedent for internal control counters that never leave the entity.
- `weight_rd_addr`: `std_ulogic_vector(num_bits_needed(g_weight_buffer_depth
  - 1) - 1 downto 0)` (§3.1).
- `m_accum_m2s.data` low `g_accum_width * g_pe_rows` bits: lane `l` at
  bits `g_accum_width*(l+1)-1 downto g_accum_width*l` — same convention
  `cnn_accel_bias_requant` already reads `s_accum_m2s.data` with
  (`accum_l <= signed(s_accum_m2s.data(g_accum_width*(l+1)-1 downto
  g_accum_width*l))`, `cnn_accel_bias_requant.vhd:205`); remaining high
  bits `0`. Elaboration-time assert: `g_accum_width * g_pe_rows <=
  axi_stream_data_sz` (same style as `cnn_accel_pool`'s own
  `g_accum_width <= axi_stream_data_sz` assert).

## 8. Latency/throughput

Zero-backpressure, steady state: `num_groups + 3` cycles per window
(`1` accept cycle + `num_groups + 1` run cycles + `1` cycle for
`out_valid_q` to register `'1'` after a same-cycle commit) from
`s_window` accept to `m_accum` becoming valid, **but** because the
compute engine can start window `N+1` the cycle right after accepting it
(independent of whether window `N`'s result has drained yet), sustained
throughput at zero stall on `m_accum` is one window every `num_groups + 1`
cycles (bounded by the `run` state's own length, not by the output
register) — the `+2` "idle accept" and "output register" cycles overlap
with the *previous* window's still-draining output. Example: `g_pe_cols =
4`, `3x3` kernel, `in_channels = 4` (`CONV2D`): `mac_taps = 36`,
`num_groups = 9`, so 1 window every 10 cycles at zero stall.

## 9. Corner cases covered by the verification plan (§10)

- `1x1` kernel, `in_channels = 1` (single-tap, single-group MAC — the
  `FC`-shaped degenerate case, §3.5's `CONV2D`/`FC` path with
  `mac_taps = 1`).
- `mac_taps` not a multiple of `g_pe_cols` (last group partially valid —
  exercises the `valid` mask on both `CONV2D` and `DWCONV2D`).
- `DWCONV2D` with `in_channels < g_pe_rows` (some lanes structurally
  unused/masked every group).
- Back-to-back windows with independent randomized backpressure on both
  `s_window` and `m_accum`.
- `last` propagation from the accepted window through to the emitted
  `m_accum` beat, including when a window with `last='1'` is accepted
  while the *previous* window's result is still parked in the output
  register (drains in FIFO order — only one result is ever pending, so
  this is trivially in-order).
- Dedicated zero-stall timing case confirming the §8 throughput estimate
  (`num_groups + 1` cycles/window) is actually met, not just "eventually
  correct."

## 10. AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Two independent AXI4-Stream links (`s_window` in, `m_accum` out), full
backpressure per the requirement's own "Protocols" section. `s_window_s2m.
ready` is a pure function of registered state (`state_q = idle`, never of
`s_window_m2s.valid` itself) — no combinational loop. `m_accum_m2s.valid`
is `out_valid_q`, likewise pure registered state. `weight_rd_addr`/
`weight_rd_data` is not an AXI4-Stream link (no `valid`/`ready`, matching
`cnn_accel_weight_buffer`'s own "simple read port (not AXI4-Stream:
random-access within the active bank)" characterization in
`doc/cnn_accel_arch.md`'s inter-module interface table) — a registered,
1-cycle-latency, always-ready read, driven unconditionally by this
module's own sequencing counter.

## 11. Verification plan

Hand-rolled record-port stimulus/monitor procedures (`tb_pe_array.vhd`),
same practical choice `tb_cnn_accel_pool.vhd`/`tb_cnn_accel_weight_buffer.
vhd` already made (record-typed `axi_stream_m2s_t`/`s2m_t` ports, not flat
`t*` ports the VUnit `axi_stream_master`/`slave` VCs expect). Golden
values from an independently re-derived VHDL function transliterating
`cnn_accel_model.py`'s `_conv2d_generic` MAC accumulation (not calling
into the RTL's own helper function), matching the same "no double-
rounding/no premature saturation" precedent `tb_cnn_accel_pool.vhd` set.
Test cases (entity has only `runner_cfg`; per-test stall percentages are
plain internal signals set per `run(...)` branch, following
`tb_cnn_accel_window_gen.vhd`'s `cur_stall_out` precedent — no
`module_cnn_accel.py` wiring needed this round, matching that module's/
`tb_cnn_accel_weight_buffer.vhd`'s precedent of no generic-driven stall
sweep):

1. `test_conv2d_single_window` — one `CONV2D` window, `g_pe_cols`-aligned
   and non-aligned `mac_taps`, checked against the golden MAC sum.
2. `test_dwconv2d_channel_isolation` — `DWCONV2D`, `in_channels < g_pe_rows`
   and `in_channels = g_pe_rows`; confirms each lane sums only its own
   channel's spatial taps (a deliberately-planted cross-channel weight
   value that would corrupt the result if the array ever summed across
   channels).
3. `test_fc_degenerate` — `FC`-shaped `1x1` kernel (`kernel_h=kernel_w=1`),
   `in_channels` swept, confirming it is handled by the same `CONV2D` path
   with `mac_taps = in_channels`.
4. `test_backpressure` — many windows back-to-back, independent randomized
   stall on `s_window` and `m_accum`, mixed `CONV2D`/`DWCONV2D`.
5. `test_last_propagation` — `last` on the input window correctly reaches
   the matching output beat, including the pending-output-register overlap
   case (§9).
6. `test_full_throughput` — zero stall on both links; confirms the §8
   `num_groups + 1` cycles/window steady-state bound via `check_relation`
   against `now`.

## 12. Open items / deviations recorded for the final doc

- `g_weight_buffer_depth` and `cfg_kernel_h`/`cfg_kernel_w` added beyond
  the requirement (§3.1/§3.2) — both are additive (no requirement field
  removed/contradicted).
- `cfg_out_channels` present but unused in the datapath (§3.3) —
  `out_channels > g_pe_rows` multi-tile iteration is an explicit, IP-wide
  open item, not unique to this module.
- No overlap deeper than one pending output beat (§3.6) — a genuinely
  pipelined (2+ windows in flight through the compute engine itself)
  microarchitecture is deferred, per the requirement's own "exact
  DSP-packing/pipelining microarchitecture is deferred" note and
  `doc/cnn_accel_arch.md`'s "Open items for `vhdesign`/`vhfill`" entry on
  this exact module.
