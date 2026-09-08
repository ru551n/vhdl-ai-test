# TOSA → `cnn_accel` compiler — architecture analysis and implementation plan

Status: PLAN ONLY (no implementation). Rev 2, 2026-09-07 (adds ISA extensions 1–2 as H1/H2 + M11/M12).
Canonical copy: `~/.local/state/maki/plans/smashing-patient-albacore.md`; sync to `doc/tosa_compiler_plan.md` in the repo.

Decisions already ratified by the user for this plan:

| Topic | Decision |
|---|---|
| Location / language | Python package `compiler/` in this repo; venv `.venv-compiler` (Python 3.12; iree, torch-mlir, numpy, pytest) |
| Rounding mismatch | Change `cnn_accel` RTL + golden model from round-half-even to TOSA round-half-up (separate HW milestone) |
| Tiling in MVP | Capability checking only; tiling pass designed, implemented post-MVP |
| TOSA input | Hand-written `.mlir` in MLIR **generic form** |
| Zero points / per-channel | MVP verifier rejects `zp != 0` and `per_channel`; ISA extensions 1 (output offset + general clamp) and 2 (per-channel scale) are **scheduled** (H1/H2 + M11/M12), `input_zp`/`weight_zp` remain rejected |

Empirical facts this plan is grounded in (verified in this session, see
memory `tosa_compiler_env_findings.md`):

- TOSA v1.0 generic form of `conv2d`/`rescale`/`clamp`/`const` as printed by
  `iree-opt --mlir-print-op-generic` (zero points, multiplier and shift are
  **operands**, not attributes).
- IREE compiles and executes TOSA on CPU and its `rescale` rounds **half-up**
  (`x*0.5` of `[1,-1,5,-5,3,-3]` → `[1,0,3,-2,2,-1]`). The current
  `cnn_accel_bias_requant` rounds half-to-even (`[0,0,2,-2,2,-2]`).
- torch-mlir does **not** lower PT2E quantized graphs to `tosa.rescale`;
  hand-written TOSA + IREE as oracle is the realistic MVP input path.
- `cnn_accel` (this repo): per-layer 64-byte descriptor ISA; DDR byte images
  are decision S6 channel-tiled activation planes `[ceil(C/T)][H][W][T]`
  (`T = ACTIVATION_PLANE_CHANNELS`), D10/D11 tile-major weights
  (`pack_weights_for_hw` of TOSA's logical `[OC,KH,KW,IC]`, zero-padded to
  `TILE_CHANNELS x PE_ROWS` tiles) and `PE_ROWS`-padded int32 LE bias; full
  int32 requant multiplier (33×32-bit product), combined shift `15 +
  requant_shift`, ReLU before int8 saturate, no zero points, per-tensor
  scale only, Cin/Cout tiling done **inside** the hardware, no partial-sum
  load/store. `cnn_accel_model.run_program()` is an existing ISA-level
  golden model.

---

## 1. Proposed architecture

```
 .mlir (TOSA, generic form)
        │  frontend/mlir_generic.py      (tiny generic-form parser, no MLIR dependency)
        ▼
 MlirModule (ops, operands, attrs, types)
        │  frontend/tosa_import.py       (TOSA subset → Graph IR, TOSA-level checks)
        ▼
 ┌─ Graph IR (GIR) ──────────────────── WHAT ───────────────────────────────┐
 │ passes/normalize.py        drop identity clamp, fold shape facts        │
 │ passes/legalize_rescale.py shift<15 → (mult<<k, shift+k), rounding policy│
 │ passes/fuse.py             conv2d+rescale+clamp → fused_conv (target-aware)
 └──────────────────────────────────────────────────────────────────────────┘
        │  lower/to_hir.py               (unit selection + capability check)
        ▼
 ┌─ Hardware IR (HIR) ───────────────── HOW ────────────────────────────────┐
 │ lower/tile.py       (post-MVP) split ops that exceed unit capabilities   │
 │ lower/schedule.py   order + explicit dependencies                        │
 │ lower/memplan.py    lifetimes → DDR buffers, addresses, alignment        │
 └──────────────────────────────────────────────────────────────────────────┘
        │  backend/cnn_accel_v1/emit.py  (HIR → LayerDesc list + memory image)
        ▼
 Accelerator program  = manifest.json + program.bin + constants.bin
        │
        ├── backend/cnn_accel_v1/run.py → cnn_accel_model.run_program()   (ISA simulator)
        ├── gir/interp.py                 (TOSA-semantics reference, numpy int64)
        └── IREE (iree-compile/iree-run-module)                            (external oracle)
```

Three representations, not four. "Scheduled IR" and "memory-planned IR"
are the **same** HIR data structure with more fields filled in; each stage
has its own verifier that asserts which fields must now be populated. This
avoids three near-identical class hierarchies and three conversions while
still giving a dump + verifier after every pass.

Package layout (proposed):

```
compiler/
  cnnc/
    frontend/   mlir_generic.py  tosa_import.py
    gir/        ir.py  verify.py  printer.py  interp.py
    passes/     normalize.py  legalize_rescale.py  fuse.py
    target/     contract.py  load.py          # capability model
    targets/    cnn_accel_v1.json  synthetic_tiled.json (post-MVP)
    hir/        ir.py  verify.py  printer.py
    lower/      to_hir.py  schedule.py  memplan.py  tile.py (post-MVP)
    backend/cnn_accel_v1/  emit.py  run.py
    driver.py   cli.py                        # pipeline, --dump-after-all
  tests/        (pytest; fixtures/*.mlir)
```

Rules: pure Python 3.12, `numpy` only inside `gir/interp.py`; every IR is
plain dataclasses, deterministic `dump()` text and JSON; no pass mutates its
input (returns a new IR) so dumps before/after are trivially comparable.

## 2. IR design

### 2.1 Graph IR (GIR) — WHAT

Single function, SSA, NHWC with `N == 1`, integer dtypes only.

```
Tensor  { id, shape: tuple[int], dtype: i8|i32, const: bytes|None }
Op      { id, kind, inputs: [Tensor], outputs: [Tensor], attrs }
kinds   const | conv2d | rescale | clamp | add | fused_conv
```

Quantization parameters are first-class, fully integer:

```
RescaleParams { multiplier: list[int], shift: list[int], per_channel: bool,
                in_zp: int, out_zp: int, rounding: SINGLE_ROUND|DOUBLE_ROUND|INFERENCE }
ConvAttrs     { pad: (t,b,l,r), stride: (h,w), dilation: (h,w), in_zp, w_zp, acc_type: i32 }
ClampAttrs    { min: int, max: int }
```

Example — the probe graph (`1x8x8x4 i8` → `3x3` conv, 8 out channels):

```
gir.func @main(%in: 1x8x8x4xi8) -> 1x8x8x8xi8 {
  %w    = const  : 8x3x3x4xi8   (blob 288 B)
  %b    = const  : 8xi32        (blob 32 B)
  %acc  = conv2d %in, %w, %b   {pad=[1,1,1,1] stride=[1,1] dil=[1,1] in_zp=0 w_zp=0} : 1x8x8x8xi32
  %q    = rescale %acc         {mult=[1073741824] shift=[38] per_channel=0 in_zp=0 out_zp=0 round=SINGLE} : 1x8x8x8xi8
  %y    = clamp %q             {min=0 max=127} : 1x8x8x8xi8
  return %y
}
```

After `fuse`:

```
  %y = fused_conv %in, %w, %b {conv: pad=[1,1,1,1] stride=[1,1];
                               rescale: mult=1073741824 shift=38 round=SINGLE;
                               clamp: [0,127]} : 1x8x8x8xi8
```

`fused_conv` is still WHAT: its semantics are *defined* as the composition
of the three unfused ops, and `gir/interp.py` executes it exactly that way.

### 2.2 Hardware IR (HIR) — HOW

```
Buffer   { id, space: DDR|(future: SRAM), size_bytes, align, addr: int|None,
           role: input|output|const|intermediate,
           layout: PLANES|TILED_OHWI|I32_TILED (cnn_accel S6/D10) | HWC|OHWI|I32_VEC }
HirOp    { id, unit: str, kind: str, params: dict, reads: [Buffer], writes: [Buffer],
           deps: [HirOp.id], seq: int|None }
HirModule{ target, ops, buffers, entry_inputs, entry_outputs, program: Buffer|None }
```

Same conv after `to_hir` + `schedule` + `memplan`:

```
buffers:
  %in   DDR PLANES     512 B  align 64  addr 0x00010000  role input   (4 ch -> one 8-ch plane)
  %w    DDR TILED_OHWI 576 B  align 64  addr 0x00001000  role const   (ic 4..7 zero lanes)
  %b    DDR I32_TILED   32 B  align 64  addr 0x00001280  role const
  %y    DDR PLANES     512 B  align 64  addr 0x00020000  role output
ops:
  #0 seq 0 unit conv_engine kind conv_layer
     reads [%in %w %b] writes [%y] deps []
     params { in: 8x8x4, out_ch 8, k 3x3, s 1x1, pad t1 b1 l1 r1,
              bias_en 1, requant_en 1, scale 1073741824, shift 23, relu_en 1 }
program: DDR 128 B (2 descriptors: CONV2D, HALT) addr 0x00000000
```

Note `shift 23 = 38 - 15`: the HW's implicit Q15 is absorbed at lowering,
never earlier.

### 2.3 Accelerator program

`manifest.json` (addresses, sizes, tensor descriptors, target id, compiler
version, per-op provenance back to GIR op ids) + `program.bin` (descriptor
stream) + `constants.bin` (weights/bias at their planned addresses). The
manifest is what both the simulator and a future FPGA host loader consume.

## 3. MLIR strategy

**Recommendation: standalone Python IR; MLIR is used only as the *input
format* (generic-form text) and as an *external oracle* (IREE).**

Justification:

| Criterion | MLIR dialects (C++) | Standalone Python IR |
|---|---|---|
| Implementation complexity | Out-of-tree LLVM build, TableGen, C++; no toolchain installed; hard for MCP-driven agents to iterate | Dataclasses + pytest; agents already productive in this repo's Python |
| Debugging | Excellent (`-print-ir-after-all`) but opaque build failures | `--dump-after-all` reproduces the same workflow in ~50 lines |
| Pattern rewriting / canonicalization | Best in class — but this compiler has ~5 coarse passes on graphs of 10–100 layer nodes; rewrite-engine benefits are marginal | Direct graph edits are simpler and fully inspectable |
| Shape inference | Free for TOSA | Trivial for the supported subset (conv/rescale/clamp/add), and needed anyway as a verifier |
| Testing | FileCheck/lit | pytest + golden dumps; same idea |
| Long-term | Ties the project to LLVM release churn (TOSA changed operand/attr layout in 2025) | Importer pins one TOSA version; the IR is stable regardless |

What we deliberately keep from MLIR: its **textual generic form** as the
input contract (`%n = "tosa.op"(%a, %b) <{attrs}> : (types) -> type` — a
grammar small enough for a hand-written parser), its **op naming** in GIR
(so a future MLIR dialect would be a 1:1 mapping), and **IREE** for
independent TOSA execution. If the pass pipeline ever grows to need real
rewrite infrastructure, the GIR is designed so that a `cnnc` MLIR dialect
could replace it without touching HIR or the backend.

## 4. Compute element contract

Capabilities are **data** (`targets/cnn_accel_v1.json`) validated into typed
dataclasses (`target/contract.py`). The compiler never imports RTL
knowledge; a **consistency test** asserts the JSON agrees with
`modules/cnn_accel/cnn_accel_constants.py` (PE_ROWS, TILE_CHANNELS,
WEIGHT_BUFFER_DEPTH, MAX_ROW_TILE_WORDS, ACCUM_WIDTH), which stays the
single source of truth.

```json
{
  "name": "cnn_accel_v1",
  "program_model": "layer_descriptors",
  "memory": { "spaces": { "DDR": { "size": 268435456, "align": 64 } },
              "activation_layout": "PLANES", "activation_plane_channels": 8,
              "weight_layout": "TILED_OHWI", "bias_format": "i32_le_tiled" },
  "units": [
    { "name": "conv_engine",
      "ops": ["conv2d"],
      "dtypes": { "input": "i8", "weight": "i8", "bias": "i32", "acc": "i32", "output": "i8" },
      "kernels": [[1,1],[3,3]], "strides": [[1,1],[2,2]], "dilation": [[1,1]],
      "padding": "zero", "batch": 1,
      "internal_tiling": { "cin": 8, "cout": 8 },
      "constraints": [
        { "kind": "divisible", "expr": "out_channels", "by": 8 },
        { "kind": "max", "expr": "in_width * ceil(in_channels / 8)", "value": 512 },
        { "kind": "max", "expr": "kernel_h * kernel_w * ceil(in_channels / 8)", "value": 288 },
        { "kind": "max", "expr": "in_width", "value": 65535 }
      ],
      "epilogue": {
        "bias": true,
        "rescale": { "multiplier_bits": 32, "shift_min": 15, "shift_max": 270,
                     "rounding": "half_up", "per_channel": false,
                     "input_zp": false, "output_zp": false },
        "clamp_ranges": [[-128,127],[0,127]]
      },
      "isa_version": "1.0",
      "partial_sum_io": false
    }
  ]
}
```

Design points:

- Constraint expressions use a fixed tiny grammar (`ceil`, `* / + -`,
  integer literals, layer field names); no `eval`. The tiling pass later
  uses the same constraints to decide *what* to split.
- `rounding` is a capability. Until the HW milestone (§13, H0) lands, the
  shipped JSON says `"half_even"` and the compiler **refuses** to target it
  with a clear diagnostic — the compiler never silently emits non-bit-exact
  programs.
- `internal_tiling` tells the compiler the unit tiles Cin/Cout itself;
  `partial_sum_io: false` tells the tiling pass Cin-splitting is not
  available. A future 64-MAC target changes these numbers and the
  constraint values; the frontend and GIR passes are untouched.
- `epilogue` describes what may be fused into the unit; §9 uses it. After
  H1 the JSON flips `output_zp: true`, `clamp_ranges: "any"`,
  `isa_version: "1.1"`; after H2 `per_channel: true`, `isa_version:
  "1.2"`. The backend emitter refuses to emit a field the target's
  `isa_version` does not define, so a stale JSON can never produce a
  descriptor the RTL misreads.

## 5. Accelerator programming model

Analysis of the options against the actual hardware:

| Option | Fit |
|---|---|
| Fine-grained primitives (`LOAD`, `CONV`, `RESCALE`, `CLAMP`, `STORE`) | **Fiction for this HW**: DMA, requant and ReLU are inside the layer engine, never separately addressable. Emitting them would require a sequencer that does not exist and buys nothing. |
| Compound layer descriptors (`CONV_REQUANT_RELU` + flags) | **What `cnn_accel` already is** (64-byte `LayerDesc`, `next_instr_addr`). Matches CNN structure, trivial dependency model (sequential), simulator exists. |
| Command stream / microcode | Premature; no unit can run concurrently with another yet. |
| Dependency graph in the program | Not needed while execution is strictly sequential; HIR keeps explicit `deps` so a future multi-unit target can emit them. |

**Recommendation:** the accelerator program is the existing straight-line
descriptor stream terminated by `HALT`. The compiler's HIR remains
general (explicit `deps`, `unit`, optional explicit DMA ops), so if a later
accelerator exposes standalone units (e.g. an `add` unit or SRAM-resident
tensors) the backend gains ops without changing the frontend or GIR.

ISA extensions. 1 and 2 are **scheduled in this plan** (HW milestones H1/H2,
compiler milestones M11/M12); 3 and 5 are recommendations only. All fit in
the reserved bytes W13–W15 (12 bytes) and flag bits 4–7; none is needed for
the MVP. Every field is declared once in `cnn_accel_constants.py`
(`ISA_LAYOUT`, `FLAGS`) and propagates to VHDL via hdl-registers.

1. **Output offset + general clamp** (`ISA v1.1`):
   `output_offset` (2 B signed), `clamp_min` (1 B signed), `clamp_max`
   (1 B signed) in W13; flag `CLAMP_EN` = bit 4.
   Epilogue becomes `s = round_shift(total*scale); s += output_offset;
   y = clamp(s, lo, hi)` with `lo/hi = clamp_min/clamp_max` when `CLAMP_EN`,
   else today's `lo = 0 if RELU_EN else -128, hi = 127`. Backward
   compatible: `CLAMP_EN=0` is bit-identical to current behaviour. Enables
   TOSA `output_zp` and arbitrary `clamp`. Merges the earlier "extension 4"
   because with `output_zp != 0` a TOSA ReLU is `clamp[output_zp,127]`,
   which `RELU_EN` alone cannot express.
2. **Per-channel requantization** (`ISA v1.2`): `scale_addr` (4 B) in W14;
   flag `PER_CHANNEL_EN` = bit 5. DDR table of `out_channels` entries ×
   8 B: `multiplier i32 LE, shift u8, 3 B zero` (one 64-bit AXI beat per
   channel). Loaded per output-channel tile through the same DMA path and
   alongside the bias buffer (`BIAS_BUFFER_DEPTH = PE_ROWS` entries →
   a parallel `scale_buffer` of PE_ROWS × 40 bits); `bias_requant` takes a
   per-lane `(scale, shift)` vector instead of one `cfg_requant_*`. When
   `PER_CHANNEL_EN=0` the lane vector is broadcast from the descriptor
   fields → bit-identical to today.
3. `PSUM_IN/PSUM_OUT` flags + int32 psum address: enables compiler-side Cin
   tiling and larger-than-buffer layers. Not scheduled.
5. `pad_value` (1 B signed): pad with a constant instead of literal 0. With
   `pad_value = input_zp`, TOSA `input_zp` folds **exactly** into the bias
   (`bias'[o] = bias[o] - input_zp·Σw[o]`, padded taps cancel identically
   in TOSA and HW). Cheapest path to asymmetric *inputs*; recommended
   next, not scheduled. `weight_zp != 0` stays rejected (data-dependent
   `-w_zp·Σx` term needs a datapath change).

## 6. Quantization model

Notation: TOSA as executed by IREE/spec; HW as `cnn_accel_bias_requant`.

**TOSA semantics (subset):**

```
conv2d : acc[o] = Σ (x - in_zp)(w - w_zp) + bias[o]                (i32, bias added inside conv2d)
rescale: v   = acc - in_zp_rescale                                  (must be 0 for i32 input)
         r   = 1 << (shift-1)                                       (+/- 2^30 if DOUBLE_ROUND and shift>31)
         s   = (v * mult + r) >> shift                              (i64 arithmetic, floor shift)  = round-half-up
         y   = clamp_i8(s + out_zp)
clamp  : y   = clamp(y, min_val, max_val)
```

**HW semantics (after H0 rounding change):**

```
total   = acc + bias                    (33-bit, exact)
product = total * scale                 (65-bit, exact; scale is signed int32)
s       = (product + (1 << (S-1))) >> S  with S = 15 + requant_shift   (floor shift = half-up)
s       = max(s, 0) if relu_en
y       = saturate_i8(s)
```

**Mapping rules (implemented in `legalize_rescale` + `to_hir`, verified by `gir/interp` vs `cnn_accel_model`):**

| TOSA | Condition | HW |
|---|---|---|
| `in_zp`, `w_zp` (conv) | must be 0 (MVP) | — (padding is literal 0 in HW; TOSA pads with `in_zp`, so folding `in_zp` into bias is *only* exact without padding — hence rejected) |
| bias | always | `bias_en=1`, bias buffer int32 LE |
| `multiplier` | `scale32=true`, `0 <= mult < 2^31`, per-tensor | `requant_scale = mult` (fits signed int32) |
| `shift` | `shift >= 15` | `requant_shift = shift - 15`, must be `<= shift_max - 15` |
| `shift < 15` | `mult << (15-shift) < 2^31` | `mult <<= k, shift += k` — exact for both floor and half-up rounding because the rounding constant scales identically; otherwise reject |
| `rounding_mode` | `SINGLE_ROUND` (and `INFERENCE`, which the spec lets the implementation choose; recorded in manifest) | half-up |
| `DOUBLE_ROUND` | — | reject (value-sign-dependent correction not expressible) |
| `in_zp` (rescale) | 0 (spec requires for i32) | — |
| `out_zp` | 0 (MVP); any int8 after H1/M11 | `output_offset = out_zp`, `CLAMP_EN=1`, `clamp = [-128,127]` (TOSA saturate) or the following clamp's bounds |
| `clamp [0,127]` after rescale | — | `relu_en=1`. Proof: HW `sat_i8(max(s,0)) = clamp(s,0,127)`; TOSA `clamp(clamp_i8(s),0,127) = clamp(s,0,127)`. Identical. |
| `clamp [-128,127]` | — | identity, removed by `normalize` |
| any other clamp `[lo,hi]` | reject (MVP); after H1/M11 | `CLAMP_EN=1`, `clamp_min=lo`, `clamp_max=hi`. Proof: TOSA `clamp(clamp_i8(s+zp),lo,hi) = clamp(s+zp,lo,hi)` since `[lo,hi] ⊆ [-128,127]`; HW computes exactly the right-hand side |
| `per_channel=true` | reject (MVP); after H2/M12 | `PER_CHANNEL_EN=1`, `scale_addr` → table `(mult[oc], shift[oc]-15)`; the `shift<15` legalization runs per element |
| accumulator | TOSA i32 acc incl. bias | HW int32 MAC + 33-bit bias add. Contract (D3): programs whose true accumulation leaves int32 are **invalid**; both simulators raise `AccumulatorOverflow` rather than wrap. Compile-time bound check: `9·Cin·127·128 + |bias|` worst case emitted as a warning when it could exceed 2^31. |
| signedness | all signed i8/i32 | `input_unsigned/output_unsigned` must be false |

Floating point never appears in any IR: `RescaleParams` are ints; the
importer rejects `f32` tensors; `gir/interp.py` uses `numpy.int64` with
explicit overflow assertions (products are ≤ 2^63 by construction:
33-bit × 31-bit).

**Bit-exact verification chain (each link is a test):**

```
IREE(tosa.mlir)  ==  gir.interp(GIR)  ==  gir.interp(fused GIR)  ==  cnn_accel_model.run_program(program)  ==  RTL (VUnit, vectors)
```

Tie cases are tested explicitly (values whose `product mod 2^S == 2^(S-1)`),
since that is the only place half-up vs half-even differ.

## 7. Tiling strategy (designed now, implemented post-MVP)

Where: HIR-level pass `lower/tile.py`, after unit selection, before
scheduling and memory planning — tiling creates new HIR ops and new
`BufferView`s, which the planner must see.

Why HIR and not GIR: tiling is a HOW decision driven by unit constraints;
GIR stays a faithful WHAT.

Representation:

```
BufferView { buffer, offset_elems: (h,w,c), shape: (h,w,c), contiguous: bool }
HirOp.params.tile = { axis, index, count, halo_before, halo_after, first, last }
```

Axis analysis for HWC layout + this DMA model (contiguous ranges only):

| Axis | Input slice contiguous? | Output slice contiguous? | Exactness | Needs HW |
|---|---|---|---|---|
| H (row bands) | yes (rows outermost) incl. `K-1` halo rows | yes | exact; band-internal pads set to 0, outer bands keep top/bottom pad | none → **first tiling axis** |
| Cout | n/a | **no** (C innermost) | exact | strided store or re-layout |
| Cin (accumulate) | **no** | int32 psum | exact if psum kept int32 | `PSUM_IN/OUT` |
| W | no (row stride) | no | exact | strided DMA |

Cin tiling with accumulation (the 80×80×32→40×40×64 example): tile `t`
emits `conv(in[:,:,16t:16t+16], w[..., 16t:16t+16])` with
`first=(t==0)` (zero psum), `last` (apply bias+rescale+clamp), psum in an
int32 DDR/SRAM buffer between tiles. Bias/requant only on `last`. Verified
against the untiled interpreter. Since `cnn_accel_v1` has
`partial_sum_io=false` and `internal_tiling.cin=8`, the pass never selects
this axis for it; a `synthetic_tiled.json` target (small `max` constraints,
`partial_sum_io=true`) exercises the pass in the simulator.

MVP behaviour: `tile.py` is the identity; `to_hir` raises a
`CapabilityError` listing the violated constraint when a layer does not fit.

## 8. Memory model

Distinct concepts, present from day one even though MVP uses only DDR:

```
logical tensor (GIR Tensor)  →  Buffer (HIR: space, size, align, role)  →  address (memplan)
                                 BufferView (tiled sub-range; post-MVP)
```

MVP facts: `cnn_accel` streams everything from DDR; on-chip line/weight
buffers are hardware-managed and invisible to the program. So the planner
allocates only DDR, but the `space` field exists so an SRAM-resident
target adds a space, not a redesign.

Planner (`lower/memplan.py`):

1. Lifetimes from the schedule: `[def_seq, last_use_seq]` per buffer.
2. Fixed regions in order: program (descriptors), constants (weights/bias,
   read-only, packed), entry inputs/outputs (pinned, reported in manifest).
3. Intermediates: interval-based first-fit with reuse (sorted by start;
   free list); alignment from target (`64` = safe for the 64-bit AXI
   master and any burst boundary; `4 KiB` burst rule is the DMA's job, but
   the planner also avoids placing a buffer such that it straddles the
   4 KiB address space end).
4. Output: every `Buffer.addr` set; `HirModule.memory_size`.

Verifier: no two buffers with overlapping lifetimes overlap in address;
every `addr % align == 0`; every buffer inside `DDR.size`; program region
disjoint from all data; constants match blob sizes.

Later: double-buffering and SRAM placement are additional `space`s and a
cost model; DDR external I/O layout stays as defined here.

## 9. Fusion strategy

Valuable fusions and where:

| Fusion | Value | Where | MVP |
|---|---|---|---|
| `conv2d + rescale + clamp[0,127]` → `fused_conv` | mandatory — the HW has no standalone rescale/clamp | GIR pass `fuse.py`, target-aware (`epilogue` capability) | yes |
| `conv2d + rescale` (no clamp) | same | same (`relu_en=0`) | yes |
| identity clamp removal | enables the above | `normalize.py` | yes |
| `add` (residual) into conv epilogue | needs HW | later | no |
| pool / upsample / concat | needs HW | later | no |

Semantic safety: fusion only re-groups; no parameter is recomputed (the
`15` shift absorption happens in `to_hir`, after fusion, and is itself
covered by the legalization proof in §6). The interpreter evaluates
`fused_conv` as the unfused composition, so `interp(G) == interp(fuse(G))`
is a direct test, run on random shapes/weights including tie-provoking
values. Unfusable remainders (`rescale` not consumed by a supported clamp,
standalone `clamp`) leave GIR ops that `to_hir` rejects with a diagnostic
naming the op and the missing capability.

## 10. Reference simulator

Two software models, both pre-existing or small, plus one external oracle:

1. **TOSA-semantics interpreter** (`gir/interp.py`, new): executes GIR
   (fused or not) with spec semantics from §6, numpy int64, overflow
   assertions, half-up rounding, `AccumulatorOverflow` on int32 violation.
   This is the compiler's *own* definition of correctness.
2. **ISA-level simulator** (`cnn_accel_model.run_program`, existing):
   executes the emitted program byte-for-byte as the RTL would (same
   encoder table as the RTL via `cnn_accel_constants`). Wrapped by
   `backend/cnn_accel_v1/run.py`: builds the memory image from the
   manifest, runs, extracts the output tensor. Slow (pure Python) — tests
   use small shapes; a numpy fast path may be added to the model later but
   must be cross-checked against the pure version.
3. **IREE** (external, `pytest.mark.iree`, skipped when unavailable):
   compiles the *same* `.mlir` to `llvm-cpu` and runs it — an
   implementation the project did not write.

Usage: the MVP end-to-end test compiles `conv_rescale_clamp.mlir`, runs
(1), (2), (3) on the same random int8 input and asserts byte equality.
Later the FPGA/RTL replaces or joins (2): the same `program.bin` +
`constants.bin` + input image are fed to `tb_cnn_accel_*` (vector files,
same format as `generate_vectors.py`) and, once `sequencer`/`layer_ctrl`
exist, to a full-IP VUnit testbench.

## 11. Verification strategy

Per stage (`--dump-after-all` writes `NN_<stage>.txt` + `.json`):

| Stage | Dump | Verifier invariants |
|---|---|---|
| parsed MLIR | `00_mlir.txt` | generic form only; every SSA use defined; result types match op signature |
| imported GIR | `01_gir.txt` | shapes consistent (conv output = formula), dtypes ∈ {i8,i32}, N=1, consts present for zp/mult/shift, quant params integer, `scale32` |
| normalized GIR | `02_normalize.txt` | no identity clamp; still verifies |
| legalized GIR | `03_legalize.txt` | every rescale has `shift >= 15`, mult < 2^31, rounding ∈ accepted set |
| fused GIR | `04_fuse.txt` | every `fused_conv`'s epilogue ∈ target `epilogue`; `interp` equivalence test |
| HIR (mapped) | `05_hir.txt` | every op has a `unit` whose capability list admits its params; every constraint satisfied |
| HIR (scheduled) | `06_sched.txt` | `seq` total order; every dep has smaller `seq`; every read buffer written earlier or is input/const |
| HIR (planned) | `07_memplan.txt` | §8 verifier |
| program | `08_program.txt` (decoded descriptors) | `decode(encode(d)) == d`; `next_instr_addr` chain ends in `HALT`; all addresses ∈ planned buffers; reserved bytes 0 |

End-to-end: the equality chain of §6/§10, with (a) fixed fixtures (golden
outputs committed), (b) randomized weights/inputs with fixed seeds, (c)
adversarial: tie values, saturating accumulators, `shift<15`, identity
clamp, stride 2 with odd sizes, `Cin=3` (non-multiple of 8), `Cout` not
divisible by 8 (must be rejected).

Every diagnostic is a typed exception with the GIR/HIR op id, so a failing
test says *which* invariant and *which* op.

## 12. MVP definition

Smallest compiler that turns

```
tosa.const ×N, tosa.conv2d (3x3 or 1x1, stride 1|2, zero pad, zp=0), tosa.rescale (per-tensor, SINGLE_ROUND), tosa.clamp ([0,127] or identity)
```

for **one or two chained layers**, N=1, into a `cnn_accel_v1` program that
`cnn_accel_model.run_program` executes with output byte-identical to
`gir/interp` (and IREE). Includes: generic-form parser, importer, GIR +
verifier + printer + interpreter, normalize/legalize/fuse, to_hir with
capability checks, sequential schedule, DDR memory planner with reuse,
backend emitter, CLI with dumps. Excludes: tiling, `add`, zero points,
per-channel, pooling, FPGA run, performance model.

## 13. Implementation milestones

Each is one delegable task for a Sonnet agent (`task`, `general`,
`model_tier medium`), run under `.venv-compiler/bin/python -m pytest
compiler/tests`. Order is dependency order; M0/M1 are independent of each
other, H0 is a parallel HW track.

**M0 — Package scaffold + target contract**
INPUT: this plan, `cnn_accel_constants.py`.
TASK: create `compiler/cnnc` skeleton, `target/contract.py` dataclasses,
constraint-expression evaluator (no `eval`), `targets/cnn_accel_v1.json`
(with `"rounding": "half_even"` until H0), loader with schema validation.
OUTPUT: importable package; `load_target("cnn_accel_v1")`.
ACCEPTANCE: consistency test asserts JSON values == `cnn_accel_constants`
(PE_ROWS, TILE_CHANNELS, WEIGHT_BUFFER_DEPTH, MAX_ROW_TILE_WORDS, ACCUM_WIDTH);
malformed JSON/unknown constraint kind → typed error.
TESTS: `test_target_contract.py` (load, consistency, evaluator on 6 expressions, 3 negative cases).

**M1 — MLIR generic-form parser**
INPUT: `compiler/tests/fixtures/conv_rescale_clamp.mlir` (the verified probe file, generic form).
TASK: tokenizer + recursive-descent parser for generic form: module, `func.func` with `^bb0` args, `"dialect.op"(operands) <{attrs}> : (types) -> types`, `func.return`; attribute values: ints with type suffix, bools, `array<i64: …>`, `dense<…> : tensor<…>` (splat and full lists), `#tosa.enum<X>`, type attrs (`i32`). Printer that re-emits generic form.
OUTPUT: `MlirModule` dataclasses.
ACCEPTANCE: `print(parse(f)) == normalize_ws(f)` for the fixture; parsing the pretty-printed (custom) form fails with "generic form required"; unknown attribute syntax → error with line/col.
TESTS: fixture round-trip, 8 attribute-grammar unit tests, 3 negative cases.

**M2 — TOSA importer → GIR + verifier + printer**
INPUT: M1 output.
TASK: `tosa_import.py` for `const/conv2d/rescale/clamp`; `gir/ir.py`, `verify.py`, `printer.py`. Zero points/mult/shift must resolve to `tosa.const` (else error). Reject: `f32`, N≠1, dilation≠1, `scale32=false`, unsigned flags, unknown ops.
OUTPUT: GIR with the §2.1 dump for the fixture.
ACCEPTANCE: "Import one quantized `tosa.conv2d` and verify all shape and quantization attributes" — every attr present and typed; conv output shape computed and equal to the declared type; verifier passes; each reject case raises the right typed error.
TESTS: fixture import (golden dump committed), 6 attribute assertions, 7 rejection tests, shape-formula tests for stride 1/2 with each pad.

**M3 — GIR interpreter (TOSA semantics) + IREE oracle test**
INPUT: M2.
TASK: `gir/interp.py`: conv2d (with zp, pad semantics = pad with in_zp), rescale (§6 incl. DOUBLE_ROUND for completeness), clamp; int64 + overflow assertions; `AccumulatorOverflow`.
ACCEPTANCE: rescale tie table matches IREE's observed `[1,0,3,-2,2,-1]`; fixture output equals IREE (`pytest.mark.iree`); conv identity/delta-kernel unit tests.
TESTS: 5 rescale table tests, 3 conv tests, 1 IREE comparison (random seed 0 input), 1 overflow test.

**M4 — normalize + legalize_rescale passes**
INPUT: M3.
TASK: remove identity clamp; `shift<15` legalization; rounding policy (`INFERENCE`→`SINGLE_ROUND` with manifest note, `DOUBLE_ROUND`→reject); pass framework with dump-after-each and verifier-after-each.
ACCEPTANCE: `interp(G) == interp(pass(G))` for random graphs; legalization exactness test on 10k random `(value, mult, shift<15)` triples vs unlegalized.
TESTS: as above + verifier re-run after each pass.

**M5 — fusion pass**
INPUT: M4, M0 target.
TASK: `fuse.py` producing `fused_conv` when `epilogue` admits; interpreter support for `fused_conv` as composition.
ACCEPTANCE: fixture → single `fused_conv`; `interp` equality on 50 random graphs incl. tie-provoking multipliers; `clamp [5,100]` remains unfused; graph without clamp fuses with `relu=False`.
TESTS: 4 structural + 1 randomized equivalence.

**M6 — GIR → HIR lowering with capability checks**
INPUT: M5.
TASK: `hir/ir.py`, `verify.py`, `printer.py`; `to_hir.py`: unit selection, constraint evaluation, param translation (shift−15, relu flag, pad fields), buffers with roles/layouts (weights OHWI = TOSA order, no transpose; bias i32 LE).
ACCEPTANCE: fixture → §2.2 HIR (addr None); rejections with op id for: `Cout=12`, `in_width*ceil(Cin/8)>512`, 5×5 kernel, stride 3, `zp≠0`, per-channel, `half_even` target.
TESTS: golden dump + 7 rejection tests.

**M7 — scheduling + memory planning**
INPUT: M6.
TASK: `schedule.py` (topological, deterministic), `memplan.py` (§8), verifiers.
ACCEPTANCE: two-layer graph: intermediate buffer freed and reused by a third layer's output when lifetimes allow; alignment/overlap verifier catches injected faults (tests mutate addresses).
TESTS: 3 planner cases, 4 verifier negative cases, determinism (two runs identical).

**M8 — backend `cnn_accel_v1` + CLI + MVP end-to-end**
INPUT: M7, `cnn_accel_model.py`.
TASK: `emit.py` (HIR → `LayerDesc` list via `cnn_accel_model.encode_program`, constants blob, manifest), `run.py` (memory image → `run_program` → output tensor), `driver.py`/`cli.py` (`cnnc compile x.mlir --target cnn_accel_v1 --out dir --dump-after-all`).
ACCEPTANCE: **MVP**: fixture compiled; `run.py` output == `interp` output == IREE output (byte-exact) for 3 seeds; decoded descriptors match HIR params; `08_program.txt` shows `CONV2D` + `HALT`. Note: passes only after H0 (until then the target refuses; test marked `xfail(reason="H0 pending")` so the gate is visible, not hidden).
TESTS: e2e ×3 seeds, decode round-trip, CLI smoke test producing all 9 dumps.

**M9 — two-layer + stride-2 + Cin=3 fixtures**
INPUT: M8.
TASK: add fixtures `two_layer.mlir` (16→32→32 channels, 3×3 s1 then 3×3 s2), `first_layer_cin3.mlir`; golden dumps.
ACCEPTANCE: e2e byte-exact for both; planner reuse observed in dump.
TESTS: 2 e2e + dumps.

**H0 — HW track: rounding change to half-up (parallel, HW agent)**
INPUT: §6 HW semantics.
TASK: `cnn_accel_model.round_shift_right_signed` default → half-up (keep convergent as opt-in for `truncate_round_signed` users, if any); `cnn_accel_bias_requant.vhd` `round_shift_right` → add `2^(S-1)` then arithmetic shift (removes the remainder comparator); `tb_cnn_accel_bias_requant` `test_round_to_even_ties` → half-up ties; regenerate `conv_core` vectors; run `cnn_accel.*` VUnit regression via vunit-mcp; update `cnn_accel_bias_requant.md` + this doc's §6; set `targets/cnn_accel_v1.js...
ACCEPTANCE: real vunit-mcp result all green; pytest for the model green; M8 e2e green.

**M10 — RTL cross-check via existing testbench vectors**
INPUT: M8, `generate_vectors.py` format, `tb_cnn_accel_conv_core`.
TASK: `backend/cnn_accel_v1/vectors.py` writes the compiler's program's per-layer stimulus/expected vectors in the format `tb_cnn_accel_conv_core` consumes; one VUnit config fed from a compiler-generated case.
ACCEPTANCE: VUnit test passes bit-exact against compiler-produced expected data (real vunit-mcp result).
STATUS: DONE. `test_bitexact_compiler_cases` (one config, `g_pe_rows=PE_ROWS`) covers `conv_rescale_clamp`/`first_layer_cin3`; all 5 `*conv_core*` VUnit configs pass (real vunit-mcp result), 291 compiler pytest pass / 7 IREE-skipped, and the missing/empty-`cases.txt` fail-loud path was confirmed negatively. `two_layer` (32 out channels) is deferred — its second layer exceeds `PE_ROWS=8`, and `conv_core`/this testbench support only a single output-channel tile until layer-level output-channel tiling (re-streaming the ifmap once per output-channel tile) lands as a future HW milestone (see `flow_status.md` M10).

**H1 — HW track: ISA v1.1, output offset + general clamp (after H0)**
INPUT: §5 extension 1.
TASK: `cnn_accel_constants.py`: `ISA_LAYOUT` reserved W13 → `output_offset`
(2, signed), `clamp_min` (1, signed), `clamp_max` (1, signed); `FLAGS["CLAMP_EN"] = 4`;
ISA self-consistency test still 64 B. `cnn_accel_model.py`: `LayerDesc`
fields with defaults 0, `bias_requantize_relu(..., output_offset, clamp_en,
clamp_min, clamp_max)` per §5 semantics, encoder/decoder round-trip.
`cnn_accel_bias_requant.vhd`: new `cfg_output_offset`, `cfg_clamp_en`,
`cfg_clamp_min/max` ports; offset add after the rounded shift, then
`clamp(lo,hi)` replacing `relu → saturate`; `conv_core` plumbs the ports.
`tb_cnn_accel_bias_requant`: tests `test_output_offset`,
`test_general_clamp`, `test_clamp_en_zero_is_legacy` (bit-identical to old
vectors). Regenerate `conv_core` vectors with new fields = 0 and prove
unchanged. Docs: `cnn_accel_arch.md` ISA table, `cnn_accel_bias_requant.md`.
ACCEPTANCE: real vunit-mcp `cnn_accel.*` green; pytest model green; all
pre-existing vector files byte-identical; `targets/cnn_accel_v1.json`
bumped to `isa_version 1.1`, `output_zp: true`, `clamp_ranges: "any"`.
TESTS: 3 new tb tests; model tests for offset+clamp corner cases
(`s+offset` beyond int8 both sides, `lo=hi`, `lo>hi` rejected by encoder).
STATUS: DONE (2026-09-08). `cnn_accel.*` VUnit all green (real run, incl.
`test_output_offset_after_shift`/`test_general_clamp`/
`test_clamp_en_zero_is_legacy` and the new `conv3x3_offset_clamp`
`conv_core` case); model + compiler pytest green; pre-existing vector data
byte-identical (`desc.txt` +3 zero records per case). The target is
discovered, not a JSON file: `discover.py` reports `isa_version 1.1`,
`output_zp: true`, `clamp_ranges: "any"` from the constants. Until M11 the
compiler rejected nonzero `out_zp`/general clamps with a `CapabilityError`
naming M11 (`to_hir._H1_FIELDS_LOWERING_IMPLEMENTED`, removed by M11).
Details: `modules/cnn_accel/flow_status.md` §H1.

**M11 — compiler: output_zp + general clamp (after H1)**
INPUT: M9, H1.
TASK: importer accepts `rescale.output_zp != 0`; `normalize` removes the
identity clamp only when the target lacks `clamp_ranges: "any"` (otherwise
it is harmless and kept for traceability); `fuse` accepts any clamp when
capability allows; `to_hir` emits `output_offset`, `clamp_en/min/max`
(rule: clamp bounds = following clamp if present else `[-128,127]`,
`relu_en=0`); `interp` unchanged (already spec-complete); emitter writes
new fields only for `isa_version >= 1.1`.
OUTPUT: fixtures `out_zp_relu.mlir` (`out_zp=-128`, `clamp[-128,127]`) and
`clamp_5_100.mlir`.
ACCEPTANCE: e2e byte-exact `interp == run_program == IREE` for both
fixtures ×3 seeds; the MVP fixture is emitted as `CLAMP_EN=1,[0,127]` and
a test asserts it is bit-identical to the `RELU_EN=1` encoding;
`isa_version 1.0` target + `out_zp != 0` → `CapabilityError`.
TESTS: 2 e2e, 1 rejection, 1 golden dump each.
STATUS: DONE (2026-09-08). `to_hir._clamp_params`/`_output_offset_params`
lower `out_zp` + fused clamp onto W13 (`output_offset`, `clamp_min/max`)
with `CLAMP_EN=1`, `relu_en=0` for `isa_version >= 1.1`; v1.0 targets keep
the legacy `RELU_EN` encoding and reject `out_zp != 0` / general clamps
with a named `CapabilityError`. `emit` reads the W13 fields from
`op.params` (rejects `clamp_min > clamp_max`), `decode.print_program`
shows `output_offset=`/`clamp=lo/hi` on v1.1 dumps. Fixtures
`out_zp_relu.mlir` (`out_zp=-128`, identity clamp, shift 34) and
`clamp_5_100.mlir` (clamp `[5,100]`, shift 35) generated by
`gen_fixtures.py` with gir/fused/hir/program goldens;
`compiler/tests/test_fixtures_m11.py` checks goldens, W13 fields in the
descriptor, `interp == run_program` ×3 seeds (both bounds hit) and
`== IREE` (`@pytest.mark.iree`, 14 IREE tests pass locally), v1.0
rejections, the MVP fixture emitted as `CLAMP_EN,[0,127]` whose bytes
differ from the `RELU_EN` encoding only in flags/W13 while the golden
model output is identical, and `write_conv_core_vectors` carrying the W13
fields into `desc.txt`. RTL cross-check: both M11 fixtures were added to
`module_cnn_accel._COMPILER_VECTORS_FIXTURES`, so
`test_bitexact_compiler_cases` now runs 4 cases (`flags 30`,
`output_offset -128` / `clamp 5..100` reach the RTL from the compiler's
own bytes) — real GHDL run: all passed. `pytest compiler`: 330 passed.

**H2 — HW track: ISA v1.2, per-channel requantization (after H1)**
INPUT: §5 extension 2.
TASK: `cnn_accel_constants.py`: `scale_addr` (4) in W14; `FLAGS["PER_CHANNEL_EN"] = 5`;
`SCALE_TABLE_ENTRY_BYTES = 8`. `cnn_accel_model.py`: `pack_scale_table_for_hw`,
`run_layer` reads the table when the flag is set, per-oc `(scale, shift)`
into `bias_requantize_relu`. RTL: `cnn_accel_weight_buffer` gains a
`scale_buffer` (PE_ROWS × 40 bits, filled by the same stream that fills
the bias buffer, one extra tile-load phase); `cnn_accel_bias_requant`
takes `lane_scale`/`lane_shift` vectors (broadcast from cfg when
`PER_CHANNEL_EN=0`); `conv_core` plumbs. **Dependency:** the DMA request
for the table is issued by `cnn_accel_layer_ctrl`, which is still PENDING
in `flow_status.md` — H2's RTL scope ends at `conv_core` (table presented
on the weight/bias stream by the testbench); `layer_ctrl` picks the flag
up when it is designed.
ACCEPTANCE: vunit-mcp green incl. new tests `test_per_channel_lanes`
(distinct scale/shift per lane, ties per lane) and
`test_per_channel_en_zero_is_legacy`; model pytest green; `conv_core`
vectors regenerated with a per-channel case and bit-exact; JSON →
`isa_version 1.2`, `per_channel: true`.
STATUS: DONE (2026-09-08). `cnn_accel.*` VUnit all green (real run, incl.
`test_per_channel_lanes`/`test_per_channel_en_zero_is_legacy` and the new
`conv3x3_per_channel` `conv_core` case in every `pe_rows` config); model +
compiler pytest green; pre-existing vector data byte-identical (`desc.txt`
+1 zero `scale_addr` record per case). W14 = `scale_addr`, `flags` bit5 =
`PER_CHANNEL_EN`, table entry `int32 LE multiplier | u8 shift | 3 zero
bytes`, tiled like the bias (`pack_scale_table_for_hw`). The RTL scope
ends at `conv_core` as planned: `weight_buffer` gained a third, `fill_is_
scale`-selected region with the bias region's depth/read address and the
table arrives on the existing fill stream in the bias tile-load phase;
`layer_ctrl` (PENDING) issues the DMA when it is designed. The target is
discovered, not a JSON file: `discover.py` reports `isa_version 1.2`,
`per_channel: true`. Until M12 the compiler rejects a fused per-channel
rescale with a `CapabilityError` naming M12
(`to_hir._H2_PER_CHANNEL_LOWERING_IMPLEMENTED`). Details:
`modules/cnn_accel/flow_status.md` §H2.

**M12 — compiler: per-channel rescale (after H2)**
INPUT: M11, H2.
TASK: importer accepts `per_channel=true` (multiplier/shift tensors of
length `Cout`); `legalize_rescale` per element; `to_hir` creates a
`Buffer(role=const, layout=SCALE_TABLE)` of `8·Cout` B and emits
`scale_addr`/`PER_CHANNEL_EN`; memplan places it in the constants region;
manifest lists it.
OUTPUT: fixture `per_channel.mlir` (16 channels, distinct multipliers,
some `shift<15` to exercise per-element legalization).
ACCEPTANCE: e2e byte-exact ×3 seeds vs `interp` and IREE; decoded
descriptor has the flag and a table address inside the constants region;
`isa_version 1.1` target → `CapabilityError`.
TESTS: 1 e2e, 1 rejection, planner test that the table is aligned and
non-overlapping.

**M13 (post-MVP) — tiling pass, row bands, synthetic target**
INPUT: §7, `targets/synthetic_tiled.json`.
TASK: `tile.py` H-band split with halos; `BufferView`; planner support.
ACCEPTANCE: layer violating `max_row_words` on the synthetic target compiles to k ops; simulator output equals untiled interpreter.

**M14 (post-MVP) — `tosa.add`, then Cin-tiling with psum on synthetic target; extensions 3/5 (§5) proposed to HW team.**

Ordering summary:

```
compiler: M0 M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8(xfail until H0) → M9 → M10 → M11 → M12 → M13 → M14
HW:                                              H0 ─────────────────────────→ H1 ──────→ H2
                                                  └ gates M8                   └ gates M11 └ gates M12
```

## 14. Risks and open questions

Decide early (they shape data structures):

1. **Rounding** — decided (H0). Until it lands the compiler is correctly
   *unable* to target `cnn_accel_v1`; M8 is gated on it.
2. **DDR layouts and N=1** — fixed by HW (S6 planes, D10/D11 tiled weights/
   bias); baked into HIR `layout` with the tile widths discovered from
   `cnn_accel_constants.py`, packed by `cnnc/lower/layout.py` and pinned
   against the golden model's `pack_*` functions in `tests/test_layout.py`.
   Fine for CNN inference; revisit only if the HW changes.
3. **Zero points / per-channel** — rejected in MVP; `output_zp`, general
   clamp and per-channel scale are scheduled (H1/H2, M11/M12). Remaining
   gap after M12: `input_zp != 0` (needs extension 5 `pad_value`, cheap)
   and `weight_zp != 0` (rare in practice; stays rejected). H2 also
   depends on `layer_ctrl`, which does not exist yet — the per-channel
   table's DMA is only testbench-driven at `conv_core` level until then.
4. **Int32 accumulator contract** — kept as "invalid program" (D3), detected
   at simulation, warned at compile time. Acceptable for the target
   backbone (worst case ≈ 3.7e7).
5. **Program/memory ownership between host and compiler** — the manifest
   defines absolute DDR addresses; a relocatable format (base + offsets) is
   trivial to add later but the manifest fields should be named
   `offset`/`base` from M8 to avoid churn.

Safe to postpone:

- Tiling implementation (design fixed in §7; HW does Cin/Cout internally).
- SRAM/double-buffering (needs HW that exposes on-chip storage).
- `add`, pooling, upsample, concat, detection head, NMS.
- Real exporter integration (torch-mlir can't emit `tosa.rescale`;
  ExecuTorch's TOSA serializer emits flatbuffers, not MLIR text — an
  importer from TOSA flatbuffers or a `tosa-serialize` bridge is a later
  frontend milestone).
- Performance model (`cycles_per_frame` exists; a scheduler cost model can
  reuse it).
- MLIR dialect adoption (GIR op names are already MLIR-style).

Open questions for the HW side:

- Exact `requant_shift` upper bound (`g_max_requant_shift`) for the target
  JSON `shift_max`.
- Confirm constraint formulas: `out_channels % PE_ROWS == 0`,
  `in_width*ceil(Cin/8) <= 512`, `K*K*ceil(Cin/8) <= 288`; the M0
  consistency test should be extended once `layer_ctrl` fixes them.
- Whether `TOSA INFERENCE` rounding mode should be treated as
  `SINGLE_ROUND` silently or require a CLI flag.

## 15. First implementation task (to delegate now)

**M1 — MLIR generic-form parser** (independent of HW state; M0 can run in
parallel with a second agent).

Prompt essentials for the Sonnet agent:

- Repo `~/git/vhdl-ai-test`, work only under `compiler/`. Interpreter:
  `.venv-compiler/bin/python`. Do not touch `modules/`, `run.py`,
  `requirements*.txt`.
- Create `compiler/cnnc/__init__.py`, `compiler/cnnc/frontend/__init__.py`,
  `compiler/cnnc/frontend/mlir_generic.py`, `compiler/tests/conftest.py`
  (adds `compiler/` to `sys.path`), `compiler/tests/fixtures/conv_rescale_clamp.mlir`
  with exactly this content (verified generic form):

```
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x8x8x4xi8>) -> tensor<1x8x8x8xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x8x8x4xi8>):
    %0 = "tosa.const"() <{values = dense<1> : tensor<8x3x3x4xi8>}> : () -> tensor<8x3x3x4xi8>
    %1 = "tosa.const"() <{values = dense<0> : tensor<8xi32>}> : () -> tensor<8xi32>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.conv2d"(%arg0, %0, %1, %2, %3) <{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x4xi8>, tensor<8x3x3x4xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x8xi32>
    %5 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = "tosa.rescale"(%4, %5, %6, %7, %8) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x8x8x8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x8xi8>
    %10 = "tosa.clamp"(%9) <{max_val = 127 : i8, min_val = 0 : i8, nan_mode = #tosa.nan_mode<PROPAGATE>}> : (tensor<1x8x8x8xi8>) -> tensor<1x8x8x8xi8>
    "func.return"(%10) : (tensor<1x8x8x8xi8>) -> ()
  }) : () -> ()
}) : () -> ()
```

- Deliver dataclasses `MlirModule{funcs}`, `MlirFunc{name, arg_names, arg_types, result_types, ops}`,
  `MlirOp{results: [(name, type)], name, operands: [str], attrs: dict[str, Attr]}`,
  `TensorType{shape: tuple[int,...], dtype: str}` and attribute variants
  `IntAttr(value, type)`, `BoolAttr`, `TypeAttr(name)`, `DenseArrayAttr(elem_type, values)`,
  `DenseElementsAttr(tensor_type, values | splat)`, `EnumAttr(dialect, kind, value)`.
  Dense element values are parsed as **Python ints only**; any float
  literal raises `UnsupportedLiteral`.
- Deliver `print_generic(module) -> str` re-emitting generic form.
- Tests (`compiler/tests/test_mlir_generic.py`): fixture round-trip
  (`print_generic(parse(text))` equals the fixture modulo whitespace);
  per-attribute unit tests for each `Attr` variant; negative tests: custom
  form `%0 = tosa.clamp %arg0 {...}` → `GenericFormRequired` with line
  number; undefined SSA operand → `UndefinedValue`; float dense literal →
  `UnsupportedLiteral`.
- Acceptance: `.venv-compiler/bin/python -m pytest compiler/tests -q` all
  green; report the real pytest output. No other files changed.

