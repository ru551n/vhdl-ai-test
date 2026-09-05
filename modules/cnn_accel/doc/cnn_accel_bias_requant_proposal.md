# cnn_accel_bias_requant — vhdesign proposal

Input: `modules/cnn_accel/doc/cnn_accel_bias_requant_req.md`.
Related: `doc/cnn_accel_arch.md` ("Type policy", "Interface record policy",
"Reset policy"), `modules/cnn_accel/doc/cnn_accel_weight_buffer_proposal.md`
(bias region addressing precedent, already implemented), `cnn_accel_model.py`
(`round_shift_right_signed`, `saturate_signed`, `bias_requantize_relu` --
authoritative golden semantics).

## 1. Requirements summary

Shared output-quantization stage, reused after `cnn_accel_pe_array`
(`CONV2D`/`DWCONV2D`/`FC`) and after `cnn_accel_pool`'s avg-sum path
(`bias_en=0`): `int32 accumulator -> (+ bias) -> (x requant_scale, Q15) ->
(>> requant_shift, arithmetic, rounded) -> saturate to int8 -> (optional
ReLU clamp at 0, before the int8 saturate)`. `g_pe_rows` independent lanes,
one per output-channel PE row. `cfg_requant_en='0'` bypasses scaling: bias
and ReLU still apply to the accumulator, then the low 8 bits pass through
as two's-complement (debug path).

## 2. Interface (as given by the requirement, widths resolved below)

| Port | Dir | Type |
|---|---|---|
| `clk` | in | `std_ulogic` |
| `reset` | in | `std_ulogic` |
| `cfg_bias_en`, `cfg_requant_en`, `cfg_relu_en` | in | `std_ulogic` |
| `cfg_requant_scale` | in | `std_ulogic_vector(31 downto 0)` |
| `cfg_requant_shift` | in | `std_ulogic_vector(7 downto 0)` |
| `bias_rd_addr` | out | `std_ulogic_vector(g_bias_addr_width - 1 downto 0)` |
| `bias_rd_data` | in | `std_ulogic_vector(g_accum_width*g_pe_rows - 1 downto 0)` |
| `s_accum_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` |
| `m_out_m2s`/`s2m` | out/in | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` |

Generics: `g_accum_width : positive := 32`, `g_pe_rows : positive := 8`
(both from the requirement), plus two **added** generics (this document):
`g_bias_addr_width : positive := 1` (§3) and `g_max_requant_shift : natural
:= 31` (§4). Per this IP's established convention, generics keep the `g_`
prefix and internal constants the `c_` prefix.

`s_accum_m2s.data`/`m_out_m2s.data` use only their low
`g_accum_width*g_pe_rows` / `8*g_pe_rows` bits respectively (the record's
`data` field is a fixed `axi_stream_data_sz`-bit vector per
`axi_stream_pkg`); unused high bits of `m_out_m2s.data` are driven `'0'`.
`m_out_m2s.user` is unused by this module's requirement and is driven
`(others => '0')`. `m_out_m2s.last` mirrors `s_accum_m2s.last` through the
pipeline stage (§5).

## 3. Design decision: `bias_rd_addr` addressing (flagged ambiguity)

The requirement lists `bias_rd_addr`/`bias_rd_data` with no accompanying
control for *which* bias row to present: no tile-index input, no
layer-start strobe on this module's own port list. This is a real gap --
`cnn_accel_weight_buffer` (already implemented, see its own proposal doc
§3.2 and its RTL) addresses its bias region by **row** (one row = one
output-channel tile, `g_pe_rows` lanes wide, `bias_rd_addr` width =
`num_bits_needed(g_weight_buffer_depth - 1)`), so a real design using
`out_channels > g_pe_rows` needs a new bias row per tile, and something
external to this module would have to drive that row index per beat.

**v1 decision (this document, not blocking on `cnn_accel_layer_ctrl`,
which does not exist yet):** assume `out_channels <= g_pe_rows` always,
i.e. exactly **one bias row per layer** (row 0), for every layer that uses
this module. `bias_rd_addr` is therefore driven as a constant all-zeros
output, independent of the streamed beats -- there is never a "wrong tile"
to select because there is only ever one. `bias_rd_data` is consequently
just a stable, already-valid value throughout the whole layer (assuming
`cnn_accel_layer_ctrl`, not yet designed, loads `cnn_accel_weight_buffer`'s
bias region and asserts `read_bank_sel` before starting the accumulator
stream) -- this module needs no extra logic to align a per-beat bias fetch
with a per-beat accumulator beat, which would otherwise be needed if
`bias_rd_addr` varied per beat given `cnn_accel_weight_buffer`'s bias read
port is registered (1-cycle latency).

Rationale for constant-zero over the alternatives considered:
- **Auto-incrementing a row counter once per accepted beat**, wrapping at
  a `g_bias_rows` generic: rejected for v1 because this module has no way
  to know when a "layer" (and hence the wrap point) starts or ends without
  an explicit strobe input not present on the port list; guessing that
  boundary risks silently reading the wrong tile's bias with no way for a
  test to distinguish "coincidentally correct" from "actually correct".
- **Adding a new tile-index input port**: the most correct long-term fix,
  but changes this module's port list beyond what `vharch`'s submodule
  table commits to, and the driver (`cnn_accel_layer_ctrl`) does not exist
  yet to specify what that port's timing contract would even be. Deferred
  to a future revision once `cnn_accel_layer_ctrl` is designed.

`g_bias_addr_width` (new generic) exists purely so `cnn_accel_top`'s future
wiring can size this port to match `cnn_accel_weight_buffer`'s actual
`bias_rd_addr` width (`num_bits_needed(g_weight_buffer_depth - 1)`) without
a width mismatch at the connection site; the value driven is always
all-zeros regardless of this generic.

**Explicit v1 limitation**, restated for the record: multi-tile bias
iteration (`out_channels > g_pe_rows`) is not supported by this module as
designed. Extending it requires a new tile-index input and is out of scope
here.

## 4. Design decision: runtime-variable combined shift vs. `truncate_round_signed`

The requirement's functional description explicitly allows folding the
fixed Q15 fractional shift (15) and the runtime `cfg_requant_shift` field
into "one combined shift amount... at vhdesign time" -- and the golden
model (`cnn_accel_model.py`'s `bias_requantize_relu`) does exactly that,
with a comment explaining *why*: a single combined `round_shift_right_signed`
call avoids the double-rounding error that two sequential rounding steps
(round off the Q15 fraction, then separately round off `cfg_requant_shift`)
would introduce.

This creates a real tension with `doc/cnn_accel_arch.md`'s submodule table,
which describes this module as "composing" `math.truncate_round_signed`
for the rounding step. `truncate_round_signed`'s number of removed LSBs is
fixed by its `input_width`/`result_width` **generics**, resolved at
elaboration -- but `cfg_requant_shift` is a genuine **runtime** ISA field
(`W10` of the instruction descriptor, loaded per layer), so the actual
shift amount is not known until run time. Three options were considered:

1. **Literally reuse `truncate_round_signed` for the whole combined
   shift**: impossible as a single instance, since its shift amount is
   generic-fixed, not a runtime input.
2. **Two separate rounding stages** (one `truncate_round_signed` instance
   for the fixed 15-bit Q15 shift, one more stage -- hand-written or a
   second generate-selected instance -- for the runtime `cfg_requant_shift`
   part): rejected. This is exactly the double-rounding case the
   requirement/golden model explicitly avoid; it would not agree with
   `cnn_accel_model.py` bit-for-bit for shift/value combinations that hit
   an intermediate tie.
3. **A `generate`-selected bank of `truncate_round_signed` instances**, one
   per supported `cfg_requant_shift` value 0..`g_max_requant_shift`, each
   fed the *same* full-precision product and combined-rounded in one step,
   muxed by `cfg_requant_shift`: technically achieves literal reuse and a
   single rounding step, but multiplies both `truncate_round_signed` and
   the downstream `saturate_signed` instance count by
   `(g_max_requant_shift + 1) * g_pe_rows` -- resource-explosive for even a
   modest shift range at `g_pe_rows=8`, and the requirement anyway
   authorizes folding into a single vhdesign-chosen shift implementation.

**Decision: option 4**, not literally listed above -- a hand-written
`round_shift_right` function (see the RTL's own comment) that reproduces
`truncate_round_signed`'s round-to-even algorithm (compare twice the
remainder against the divisor; on an exact tie, round to the quotient's
even value) but takes the combined shift amount as a plain runtime
`natural`, using `ieee.numeric_std`'s dynamic `shift_right`/`shift_left`
(both accept a runtime `natural` count and synthesize as barrel shifters --
this is exactly the documented, standard technique for a
compile-time-unknown shift amount). This is a deliberate, documented
**deviation** from the arch table's literal wording ("composing... reused
primitives"): `math.saturate_signed` genuinely *is* reused (one instance
per lane, `input_width => c_product_width, result_width => 8`, both fixed
regardless of `cfg_requant_shift`'s runtime value, so no width-matching
problem there); `math.truncate_round_signed` is not instantiated, for the
reasons above. The hand-written function is verified bit-exact against
`cnn_accel_model.py`'s `round_shift_right_signed` by the testbench (not
against the RTL's own structure -- the testbench's reference function uses
`numeric_std`'s `mod`/`/` operators, a structurally different derivation of
the same round-to-even rule, as an independent cross-check).

`g_max_requant_shift` (new generic, default 31) bounds `cfg_requant_shift`
purely as a defensive elaboration-time sizing input to the internal
product/remainder arithmetic (guarded by an `assert` in the RTL); values of
`cfg_requant_shift` above it are clamped, not rejected, so an
out-of-range configuration degrades to reduced precision rather than
undefined behavior. Compiled programs are not expected to exceed a modest
shift (the ISA field is 8 bits wide but real quantization shifts are
typically <32); this bound is not exercised by the testbench's checks
(tests stay within it) since clamping would disagree with the unbounded
Python model by construction.

## 5. Architecture and dataflow

Fully combinational per-lane compute (`accum + bias -> multiply -> round
-> ReLU-clamp -> saturate`, and separately the bypass low-8-bits path),
muxed by `cfg_requant_en`, feeding a single **flow-through output
register** (not a full skid buffer): a 1-entry `out_valid_q`/`out_data_q`/
`out_last_q` register that accepts a new input beat whenever it is empty
*or* being drained by `m_out_s2m.ready` this same cycle
(`accept_input <= (not out_valid_q) or m_out_s2m.ready`), so it sustains
one accepted beat per cycle at zero stall (full throughput) while still
giving every stateful element (just this one register) a real,
reset-clearable state, per `doc/cnn_accel_arch.md`'s reset policy. `reset`
clears only `out_valid_q` (bubble content has no completeness contract, as
elsewhere in this IP). Latency: 1 cycle, `s_accum` beat accepted at cycle
`N` appears on `m_out` at cycle `N+1` (if `m_out_s2m.ready` was high) or
whenever it next drains.

`cfg_bias_en`/`cfg_requant_en`/`cfg_relu_en`/`cfg_requant_scale`/
`cfg_requant_shift` are sampled combinationally at whatever value they hold
when a beat is accepted (consistent with the inferred assumption -- see
`cnn_accel_bias_requant_design` design-phase notes -- that
`cnn_accel_layer_ctrl` holds them stable, level not pulsed, for a whole
layer's stream; no separate latch-enable port is needed).

Per lane `l` (0 to `g_pe_rows - 1`):
- `accum_l`, `bias_l`: `g_accum_width`-bit slices of `s_accum_m2s.data` /
  `bias_rd_data` (lane 0 = low bits, ascending).
- `total_l = accum_l + (bias_l when cfg_bias_en else 0)`, in
  `c_sum_width = g_accum_width + 1` bits (guard bit, no overflow).
- Requant path (`cfg_requant_en='1'`): `product_l = total_l *
  cfg_requant_scale` (exact `c_product_width = c_sum_width + 32` bits, no
  truncation before rounding) -> `round_shift_right(product_l,
  combined_shift)` (§4) -> ReLU-clamp to 0 if negative (before saturate) ->
  `math.saturate_signed(c_product_width -> 8)`.
- Bypass path (`cfg_requant_en='0'`): ReLU-clamp `total_l` to 0 if negative
  (same ordering as the requant path, applied to the pre-scale value per
  the requirement), then take its low 8 bits as a two's-complement value
  (`signed(std_ulogic_vector(total_l)(7 downto 0))`).
- `final_lane_l = requant result when cfg_requant_en='1' else bypass
  result`.

## 6. Numeric types and widths

All arithmetic in `signed`/`unsigned` (`ieee.numeric_std`); `std_ulogic`/
`std_ulogic_vector` only at port/record boundaries, per the arch doc's type
policy. Explicit `resize` at every width-changing step; no implicit
truncation. `c_sum_width`/`c_product_width` are computed, not guessed, from
`g_accum_width` (see §2/§5). An `assert ... severity failure` guards
`g_accum_width*g_pe_rows <= axi_stream_data_sz` (§7), `8*g_pe_rows <=
axi_stream_data_sz`, `g_accum_width >= 8`, and `(15 + g_max_requant_shift)
<= c_product_width - 2`.

## 7. Cross-module width constraint (flagged, not resolved here)

`axi_stream_pkg.axi_stream_m2s_t.data` is a **fixed** `axi_stream_data_sz =
128`-bit vector (a package constant, not a generic) -- it is not resizable
per-instance. At the arch doc's own top-level defaults (`g_accum_width =
32`, `g_pe_rows = 8`), `s_accum_m2s`'s intended data width
(`g_accum_width*g_pe_rows = 256`) **exceeds** this fixed 128-bit field.
This module's generics support any combination with
`g_accum_width*g_pe_rows <= 128` (guarded by the `assert` in §6); at the
top-level defaults, `cnn_accel_top`'s own future vhdesign will need to
either narrow `g_accum_width` for this link, split the accumulator stream
into multiple beats per tile, or widen `axi_stream_pkg` (out of reach here
-- vendored, read-only reuse). Flagged as an open cross-module integration
item, not blocking this module's own design/verification, which use a
generic configuration within the fixed-width budget (see the testbench).

## 8. Corner cases

- Round-to-even ties on both even and odd quotients, both signs.
- Saturation both directions (large positive/negative pre-saturate value).
- ReLU-before-saturate: a would-have-saturated-negative value clamps to 0
  (not -128); a would-have-saturated-positive value still saturates to
  +127 (ReLU does not cap the upper range).
- All 8 combinations of `cfg_bias_en`/`cfg_requant_en`/`cfg_relu_en`.
- `cfg_requant_en='0'` bypass: two's-complement wraparound of the low 8
  bits (not saturation), bias/ReLU still applied to the pre-scale value.
- `bias_rd_addr` stays constant all-zeros regardless of streamed traffic
  (§3).
- Full-throughput sustaining (zero stall both sides) and randomized
  backpressure on both `s_accum`/`m_out` handshakes.

## 9. Verification plan

VUnit-5 testbench, `tb_cnn_accel_bias_requant` (VHDL-2008), hand-rolled
push/pop procedures directly against the `axi_stream_m2s_t`/`s2m_t` record
signals (matching `tb_cnn_accel_weight_buffer`'s established precedent in
this IP, avoiding VUnit's raw `axi_stream_master`/`slave` verification
components' `std_logic`-typed ports, which would need an extra bridging
layer against this module's `std_ulogic` record ports for no behavioral
benefit). Expected values computed by a testbench-local reference function
independently transliterated from `cnn_accel_model.py`'s
`round_shift_right_signed`/`saturate_signed`/`bias_requantize_relu`
(structurally different derivation -- `numeric_std`'s `mod`/`/` operators
rather than the RTL's shift-and-subtract remainder -- so agreement between
RTL and testbench is a real cross-check, not a tautology), not against the
RTL's own internal signals. `stall_probability_percent : natural := 0`
generic (module_canny.py precedent) drives both a per-beat randomized
input-stall and a per-cycle randomized `m_out_s2m.ready` Bernoulli
toggle. Test cases: round-to-even ties, saturate clamping (both
directions), ReLU-before-saturate ordering, the 8 bias/requant/relu flag
combinations, the `cfg_requant_en='0'` bypass path, one dedicated
zero-stall full-throughput timing case, and one randomized-backpressure
case (forces a non-zero stall locally regardless of the generic's default,
so a standalone `vunit-mcp` run without `module_cnn_accel.py`'s future
per-test generic override still exercises real backpressure). See the
testbench file itself and this project's final report for the exact test
names and recommended per-test `stall_probability_percent` values for
`module_cnn_accel.py`'s `setup_vunit`.

## Implementation Notes (vhfill)

Implemented as designed above; no proposal changes were needed during
`vhfill`. `vunit-mcp` (`nvc`) used for compile/run; see the project's final
report for the literal pass/fail summary. No unresolved `--@` markers
remain in `cnn_accel_bias_requant.vhd`.

Two bugs found and fixed only by actually compiling/running (not visible
from a read-through):

- `nvc` rejects slicing a type conversion's result directly (`the prefix
  of a slice name must be a name or a function call`), which both the
  RTL's `bypass_proc` and the testbench's `ref_bias_requantize_relu`
  originally did
  (`signed(std_ulogic_vector(x)(7 downto 0))`). Fixed in both files by
  binding the conversion to an intermediate variable first, then slicing
  that variable.
- The testbench's `bias_rd_data` is `g_accum_width*g_pe_rows` bits wide
  (not `axi_stream_data_sz`-padded like an AXI4-Stream `data` field); an
  early draft reused the AXI-Stream-data packing helper for it, which
  produced a 128-bit value against an 80-bit signal (`c_accum_width=20,
  c_pe_rows=4`) and failed at elaboration ("value length 128 does not
  match signal length 80"). Fixed with a dedicated `pack_lanes_bias`
  helper sized exactly to the port's real width.

`test_full_throughput` fails under a plain `vunit-mcp` run with this
testbench's default generics (`stall_probability_percent_in/out := 20`):
confirmed pre-existing and not specific to this module --
`cnn_accel.tb_cnn_accel_pool.test_full_throughput` fails the same way
under the same conditions, because `module_cnn_accel.py` has no
`setup_vunit` yet (its own docstring says so) and so no test gets the
zero-stall override `module_canny.py`'s precedent would give it. Manually
confirmed the RTL and testbench are both correct by invoking `nvc`
directly with `-gstall_probability_percent_in=0
-gstall_probability_percent_out=0` (bypassing `module_cnn_accel.py`,
which this project must not modify): the same 300-beat stimulus finishes
cleanly at 3025 ns, under the test's own `320 * clk_period = 3200 ns`
threshold, with zero failing checks. See the project's final report for
the recommended per-test generics for `module_cnn_accel.py`'s future
`setup_vunit` (same `0 if "full_throughput" in test.name else 20` shape
as `module_canny.py`).
