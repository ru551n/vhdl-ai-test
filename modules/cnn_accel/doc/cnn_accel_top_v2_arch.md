# CNN Accelerator Top Level — Architecture rev 2 (`cnn_accel_top`, ISA v2.0)

Supersedes the top-level portion of `doc/cnn_accel_arch.md` rev 1
(`cnn_accel_top_req.md`, `cnn_accel_sequencer_req.md`,
`cnn_accel_layer_ctrl_req.md`, `cnn_accel_csr_req.md`). The leaf compute
elements (`cnn_accel_conv_core` and its children, `cnn_accel_pool`) are
**unchanged and reused verbatim**; this document only defines the
command-driven shell around them.

Authoritative on conflict with rev 1 for: opcode table, descriptor
layout, storage model, top-level ports, CSR map, error policy.

---

## 1 Why rev 2 — the residency problem with rev 1

Rev 1's ISA is **DDR-centric**: every descriptor names `in_addr`,
`out_addr`, `weight_addr`, `bias_addr` as DDR byte addresses, so the only
possible dataflow is

```
DDR -> CONV -> DDR -> CONV -> DDR -> ...
```

Every intermediate activation tensor makes a full round trip through
external memory. For YOLOv8n at 640x640 that is the 202.3 MB / 158 ms
bandwidth figure in `YOLOV8N_GAP_ANALYSIS.md` §14 (M2), of which 184 MB is
activation re-streaming. It also makes residual/skip tensors and
concatenation impossible to express without spilling.

Rev 2 introduces an **on-chip tensor scratchpad** and makes every
descriptor operand carry a **storage-space tag**, so the natural dataflow
becomes

```
DDR -> LOCAL -> COMPUTE -> LOCAL -> COMPUTE -> LOCAL -> ... -> DDR
```

with DDR used only as backing storage: initial inputs, weights, final
outputs, and *explicit* compiler-decided spill/reload.

**Compatibility:** rev 1's `W0[31:16]` was `reserved, must be 0`. Rev 2
uses that field for the space tags, and space tag `0` is `DDR`. Therefore
**every legal ISA v1.2 program is a legal ISA v2.0 program with identical
semantics** — the compiler's existing emitter keeps working, and the
`DDR -> COMPUTE -> DDR` path remains available (it is what the `space=DDR`
encoding means).

---

## 2 Top-level architecture

```
                        external DDR4 (AXI4, 64-bit)
                                  |
      +---------------------------+---------------------------+
      |                  cnn_accel_axi_mux                    |
      |   (axi.axi_simple_read_crossbar 3:1 + write passthru) |
      +---+-------------+-------------------+-----------------+
          |             |                   |
   instr fetch    load / weight-fill    store / spill
          |             |                   |
          v             v                   v
  +---------------+  +----------------------------+  +------------------+
  | cnn_accel_    |  | cnn_accel_axi_read_dma     |  | cnn_accel_       |
  | cmd_fetch     |  | (reused, unmodified)       |  | ofmap_dma        |
  | (descriptor   |  +-------------+--------------+  | (reused)         |
  |  assembler)   |                | AXI-Stream      +---------^--------+
  +-------+-------+                v                           | AXI-Stream
          | layer_desc   +--------------------------------------+-------+
          v              |      cnn_accel_tensor_mem                    |
  +---------------+      |  banked scratchpad, byte-addressed,          |
  | cnn_accel_    |      |  g_tensor_banks x g_tensor_bank_words x 64b  |
  | cmd_proc      |      |  ports: 1 stream-write + 2 stream-read       |
  | decode /      |      |         per bank, fixed-priority arbiter     |
  | validate /    |      +---+----------------+---------------------+---+
  | dispatch /    |          | rd0            | rd1                 ^ wr
  | retire        |          v                v                     |
  +-------+-------+   +-------------+  +----------------+           |
          | start/done|             |  |                |           |
          +---------->| cnn_accel_  |  | cnn_accel_     |-----------+
          |           | conv_core   |  | elementwise    |
          |           | (REUSED:    |  | (new, H5/H1/H3)|
          |           |  window_gen |  |  ADD / LUT /   |
          |           |  pe_array   |  |  UPSAMPLE /    |
          |           |  weight_buf |  |  COPY          |
          |           |  bias_req)  |  +----------------+
          |           +------+------+
          |                  | (pool path)
          |           +------v------+
          |           | cnn_accel_  |
          |           | pool (REUSED)|
          |           +-------------+
          v
  +---------------+
  | cnn_accel_csr |  AXI4-Lite <- host/testbench
  | + counters    |
  +---------------+
```

`cnn_accel_cmd_proc` replaces rev 1's `sequencer` + `layer_ctrl` pair: one
FSM owns program-counter, decode, validation, operand-space resolution,
engine dispatch and the output-channel-tile (OT) loop.

### 2.1 Submodules

| Module | Status | Role |
|---|---|---|
| `cnn_accel_conv_core` | **reused unmodified** | CONV2D/FC datapath incl. bias+requant+clamp epilogue |
| `cnn_accel_pool` | **reused unmodified** | POOL_MAX / POOL_AVG |
| `cnn_accel_axi_read_dma` | **reused unmodified** | DDR -> AXI-Stream (instr, tensor load, weight fill) |
| `cnn_accel_ofmap_dma` | **reused unmodified** | AXI-Stream -> DDR (store/spill) |
| `cnn_accel_tensor_mem` | **new** | local tensor scratchpad (§4) |
| `cnn_accel_cmd_fetch` | **new** | 64-byte descriptor assembler from a byte stream |
| `cnn_accel_cmd_proc` | **new** | decode / validate / dispatch / OT loop / retire |
| `cnn_accel_elementwise` | **new** | ADD (H5), activation LUT (H1), UPSAMPLE (H3), COPY |
| `cnn_accel_csr` | **new** | AXI4-Lite control/status + performance counters |
| `cnn_accel_top` | **new** | structural integration only |

---

## 3 Storage model

Two-bit space tag per operand:

| Tag | Name | Address meaning | Backing |
|---|---|---|---|
| `0` | `DDR` | 32-bit external byte address | external DDR4 via AXI4 |
| `1` | `LOCAL_TENSOR` | byte offset into `cnn_accel_tensor_mem` | on-chip BRAM scratchpad |
| `2` | `LOCAL_WEIGHT` | entry index into `cnn_accel_weight_buffer` | on-chip weight/bias/scale buffers |
| `3` | — | reserved, rejected by validation (`ERR_BAD_SPACE`) | — |

Rules:

* Any operand of any command may be `DDR` or `LOCAL_TENSOR`.
* `LOCAL_WEIGHT` is legal only for the `weight`/`bias`/`scale` operands.
* `LOCAL_TENSOR` addresses are **byte offsets**, must be 8-byte aligned
  (`ERR_MISALIGNED` otherwise), and must satisfy
  `addr + length <= g_tensor_bytes` (`ERR_LOCAL_RANGE` otherwise).
* `DDR` addresses must be 8-byte aligned and within
  `g_ddr_limit` (`ERR_DDR_RANGE`).
* A tensor's activation byte layout is **identical in both spaces** — the
  PLANES packing of `cnn_accel_model.pack_activation_planes`
  (`[plane][y][x][ch_in_tile]`, 8 channels/plane, 8 bytes/word). Moving a
  tensor between spaces is therefore a pure byte copy with no repacking,
  which is what makes spill/reload trivially bit-exact.

### 3.1 Local buffer identity and lifetime

Hardware has **no buffer table and no liveness tracking**. A "local
buffer" is nothing but a `(base, length)` byte range chosen by the
compiler/Python planner. Consequences, all of them required by the
residency policy:

* Scratchpad contents are **never** invalidated by command completion.
* Contents survive across any number of commands, and across `HALT`.
* Contents are destroyed only by: an explicit write to that range, or
  `reset`/`soft_reset` (which does *not* clear the RAM but leaves it
  undefined-by-contract; tests must not rely on data surviving a reset).
* Buffer reuse, aliasing legality, and spill decisions are entirely the
  producer's (compiler's) responsibility. Hardware only range-checks.

This is the mechanism the compiler's future lifetime analysis /
allocation / spill passes need, and nothing more.

---

## 4 `cnn_accel_tensor_mem` — local tensor memory

| Generic | Default | Meaning |
|---|---|---|
| `g_num_banks` | 2 | address-interleaved on the **high** bits (a buffer stays inside one bank) |
| `g_bank_words` | 1024 | 64-bit words per bank (2 x 1024 x 8 B = 16 KiB default; tests override) |
| `g_data_width` | 64 | fixed = AXI data width |

`g_tensor_bytes = g_num_banks * g_bank_words * 8`.
Bank select = `addr[log2(bank_words*8) + log2(num_banks) - 1 : log2(bank_words*8)]`.

Each bank is one **simple-dual-port** RAM (1 write + 1 read per cycle,
`ram_style` block) — the shape Vivado/Yosys infer as RAMB36 without
hints, per the M7 lesson in `flow_status.md`.

Front-end ports: **2 write channels (`w0`/`w1`) and 2 read channels
(`r0`/`r1`)**, all AXI-Stream `axi_stream_m2s_t`/`s2m_t`, 64-bit, each
fronted by its own `dma_req_m2s_t`/`dma_req_s2m_t` request port.

| Channel | Dedicated to |
|---|---|
| `w0` | `dma_load` — DDR -> LOCAL (`LOAD`, `LOADW`, reload) |
| `w1` | `engine_out` — compute result -> LOCAL |
| `r0` | `engine_in_a`, and `dma_store` (LOCAL -> DDR); muxed by `cmd_proc` |
| `r1` | `engine_in_b` (the second `ADD` source) |

Each channel is *dedicated*, not arbitrated at the front end. That is a
deliberate correction to an earlier revision of this section, which had a
single shared write port with fixed priority between `engine_out` and
`dma_load` — that contradicted this section's own claim below that bank
independence permits ping-pong double buffering, because two requesters
sharing one front-end port serialize with each other no matter which
banks they address. Two dedicated write channels cost only a per-bank
arbiter and make the claim true.

A read/write *job* is a linear burst: `(base, length_bytes)` presented on
a `dma_req_m2s_t`-shaped request port, then `length/8` beats stream with
`last` on the final beat. This is byte-for-byte the same stream contract
`cnn_accel_axi_read_dma`/`cnn_accel_ofmap_dma` already produce/consume, so
engines are agnostic to whether their data came from DDR or the
scratchpad.

Contention is resolved **per bank, per direction**: each bank has one
physical write port and one physical read port, so two channels aimed at
different banks both proceed at full rate in the same cycle, and two
aimed at the same bank are resolved by a round-robin arbiter (one
"who-won-last" bit) that back-pressures the loser. Round-robin rather
than fixed priority so neither channel can be starved by a long burst on
the other. Because banks are independent, `dma_load` into bank 1 proceeds
concurrently with `engine_in_a`/`engine_out` on bank 0 — the architecture
does not prevent ping-pong double buffering, even though the rev-2
`cmd_proc` issues commands sequentially (see §12 limitations).

A request that crosses a bank boundary is a caller bug. The memory never
corrupts the neighbouring bank: it asserts (severity `error`, so the run
continues and the bug is visible in the log) and clamps the burst to what
fits in the addressed bank. An out-of-range bank index is likewise
asserted and then wrapped modulo `g_num_banks`, so no array bound is ever
violated.

A zero-length request (`length = 0`, i.e. zero beats) is likewise a
caller bug and is asserted the same way, on all four channels
(`w0`/`w1`/`r0`/`r1`). It must never be sent: the write channels pulse
`done` for it anyway (a zero-beat transfer trivially "completes"), but a
zero-length *read* has no beat to carry `last` on, so `done` never
pulses and a requester waiting on it would hang forever. Every producer
in this design (`cnn_accel_cmd_proc`, and `cnn_accel_elementwise` for the
`COPY`/`ACT`/`ADD`/`UPSAMPLE` opcodes it drives) validates its own
transfer length against zero before ever issuing a request here, so this
assertion is a last-resort guard against a new producer bug, not the
first line of defense.

---

## 5 ISA v2.0 — command/program format

Fixed 64-byte (16 x 32-bit little-endian word) descriptor, 8-byte-aligned,
byte-addressed, chained by `next_instr_addr`. `c_isa_version = 0x0200`,
readable from `CSR.HW_INFO2`.

### 5.1 Word layout

| Word | Bits | Field | Notes |
|---|---|---|---|
| W0 | [7:0] | `opcode` | §5.2 |
| W0 | [15:8] | `flags` | bit0 `relu_en`, bit1 `bias_en`, bit2 `requant_en`, bit3 `pad_en`, bit4 `clamp_en`, bit5 `per_channel_en`, bit6 `act_lut_en` (**v2.0**, H1), bit7 `weight_reuse` (**v2.0**) |
| W0 | [17:16] | `space_src0` | **v2.0**; 0=DDR (v1.2-compatible) |
| W0 | [19:18] | `space_src1` | **v2.0** |
| W0 | [21:20] | `space_dst` | **v2.0** |
| W0 | [23:22] | `space_wgt` | **v2.0**; applies to `weight_addr`, `bias_addr`, `scale_addr` together |
| W0 | [31:24] | reserved | must be 0 (`ERR_BAD_RESERVED`) |
| W1 | [31:0] | `in_addr` / `src0_addr` | space = `space_src0` |
| W2 | [31:0] | `out_addr` / `dst_addr` | space = `space_dst` |
| W3 | [31:0] | `weight_addr` | space = `space_wgt`; for `ACT` this is the 256-byte LUT's address |
| W4 | [31:0] | `bias_addr` | space = `space_wgt` |
| W5 | [15:0] / [31:16] | `in_width` / `in_height` | |
| W6 | [15:0] / [31:16] | `in_channels` / `out_channels` | |
| W7 | [7:0] x4 | `kernel_h`, `kernel_w`, `stride_h`, `stride_w` | |
| W8 | [7:0] x4 | `pad_top`, `pad_bottom`, `pad_left`, `pad_right` | |
| W9 | [31:0] | `requant_scale` | signed Q15 |
| W10 | [7:0] | `requant_shift` | |
| W11 | [7:0] x4 | `pool_kernel_h/w`, `pool_stride_h/w` | |
| W12 | [31:0] | `next_instr_addr` | DDR byte address; always DDR space |
| W13 | [15:0] | `output_offset` | signed int16 |
| W13 | [23:16] / [31:24] | `clamp_min` / `clamp_max` | signed int8 |
| W14 | [31:0] | `scale_addr` | space = `space_wgt` |
| W15 | [31:0] | `xfer_bytes` / `src1_addr` | **v2.0**: byte count for `LOAD`/`STORE`/`COPY`/`ACT`; second source address for `ADD` (space = `space_src1`); must be 0 for v1.2 opcodes |

`W15` was `reserved, must be 0` in v1.2 and is only consulted by the
v2.0-only opcodes, preserving compatibility.

**This field table is authoritative over §5.2's prose.** `W15` is a union,
and for `ADD` it is `src1_addr`, *not* a byte count: `ADD` has two source
operands and only one spare word to address the second one, so its length
must come from somewhere else. It comes from the tensor geometry
(`in_width` x `in_height` x `in_channels`, in the channel-tiled byte count
of §6), exactly as it does for `UPSAMPLE`. An earlier revision of §5.2
described `ADD` as "`xfer_bytes` long", which is unimplementable alongside
`src1_addr` occupying the same bits; that wording has been corrected below.

`ADD` also carries only **one** `(requant_scale, requant_shift)` pair, which
is applied identically to both operands before the sum. Two residual
branches with genuinely different scales must therefore be equalised by the
compiler (fold the ratio into the producing convolution's requant), which is
what a quantized residual add needs anyway.

`ACT` needs a 256-entry LUT address and has no dedicated field for one, so
it reuses `weight_addr`/`space_wgt` — the same "compile-time side table in
local weight memory" role `scale_addr` plays for `CONV2D`. `ACT` keeps its
`xfer_bytes` byte count, since it is a single-source streaming op.

### 5.2 Opcodes

| Code | Name | Semantics | Engine |
|---|---|---|---|
| `0x00` | `HALT` | end of program; pulse `done` | — |
| `0x01` | `CONV2D` | `dst = epilogue(conv(src0, weights))` | `conv_core` |
| `0x02` | `DWCONV2D` | allocated, **rejected** (`ERR_UNSUPPORTED_OP`) | — |
| `0x03` | `POOL_MAX` | max pool | `window_gen`+`pool` |
| `0x04` | `POOL_AVG` | average pool (divide via `bias_requant`) | `window_gen`+`pool`+`bias_requant` |
| `0x05` | `FC` | degenerate 1x1 CONV2D | `conv_core` |
| `0x10` | `LOAD` | `xfer_bytes` from `src0` to `dst`; intended DDR->LOCAL_TENSOR | DMA |
| `0x11` | `STORE` | `xfer_bytes` from `src0` to `dst`; intended LOCAL_TENSOR->DDR (this is the **spill**) | DMA |
| `0x12` | `LOADW` | `xfer_bytes` from `src0` into `LOCAL_WEIGHT`; `flags.bias_en`/`per_channel_en` select the weight / bias / scale sub-region | DMA + `weight_buffer` |
| `0x13` | `ADD` | `dst = sat_i8(requant(src0) + requant(src1))`, elementwise; `src1` from W15, length from geometry (see §5.1) | `elementwise` |
| `0x14` | `UPSAMPLE` | nearest-neighbour 2x2 replicate, `in_width`/`in_height`/`in_channels` | `elementwise` |
| `0x15` | `COPY` | local-to-local byte copy, `xfer_bytes` long | `elementwise` |
| `0x16` | `ACT` | standalone 256-entry int8->int8 LUT (at `weight_addr`) over `xfer_bytes` | `elementwise` |

`LOAD`, `STORE` and `COPY` are one opcode family differing only in the
space tags of their operands — the hardware does not care which direction
it is; `LOAD` vs `STORE` naming exists so that programs, traces and the
performance counters are self-documenting. A "reload" is simply a `LOAD`
whose source is the DDR address a previous `STORE` wrote.

### 5.3 Fusion rule

The epilogue is **not** a separate command. `CONV2D` with
`bias_en|requant_en|clamp_en|relu_en|per_channel_en|act_lut_en` performs
bias add, requantization, output offset, clamp/ReLU and LUT activation
inside `cnn_accel_conv_core`'s pipeline, and **never materializes the
int32 accumulator tensor**. A standalone `ACT`/requant command exists only
for graphs where the activation does not follow a conv.

### 5.4 Output-channel tiling

`out_channels > g_pe_rows` is handled by `cmd_proc`'s **OT loop**: one
descriptor is executed as `ceil(out_channels / g_pe_rows)` passes. Per
pass it re-fills the weight/bias/scale buffers for that tile and
re-streams the whole ifmap, writing the pass's `g_pe_rows` output planes
at `dst + ot * plane_bytes`. This is transparent to the program (one
descriptor, `out_channels` as written) and is the loop the gap analysis
flagged as never yet exercised (§24.1 finding 8).

---

## 6 External DDR memory map

The DUT discovers everything from memory; the host supplies only
`PROGRAM_BASE_ADDR` + `START`. Regions are **soft** (defined by the
program, not by hardware decode), and the Python generator lays them out
as:

| Offset | Region | Contents |
|---|---|---|
| `0x0000_0000` | reserved / null guard | never accessed; catches null-address bugs |
| `0x0000_1000` | program | chained 64-byte descriptors, `HALT` last |
| `0x0001_0000` | weights | `pack_weights_for_hw` images, per layer |
| `0x0002_0000` | bias | `pack_bias_for_hw` images |
| `0x0003_0000` | scale tables | `pack_scale_table_for_hw` images |
| `0x0004_0000` | LUT tables | 256-byte int8->int8 activation tables |
| `0x0008_0000` | inputs | initial activation tensors (PLANES layout) |
| `0x000C_0000` | spill arena | destinations of `STORE`-as-spill |
| `0x0010_0000` | outputs | final results; the region exported to CSV |

`g_ddr_limit` (default `0x0020_0000` = 2 MiB in simulation) bounds
validation. The VUnit memory model is sized to match.

---

## 7 CSV memory-image format

**One** format for all external-memory data, deliberately minimal:

```
# cnn_accel memory image v1
# <free-form provenance comments>
address,data
00001000,0000000000000001
00001008,0000000000001000
...
```

* One record per **64-bit word**; `address` = 8-byte-aligned hex byte
  address (8 hex digits, no `0x`); `data` = 16 hex digits, the little-endian
  64-bit word value as it appears on AXI `RDATA`/`WDATA`.
* Sparse and order-independent: unlisted words are undefined (the memory
  model leaves them at their default and marks them unwritten).
* Lines starting `#` and blank lines are comments. A header line
  `address,data` is required and skipped.
* The **same** parser/writer serves program, weights, bias, scale, LUT,
  inputs and outputs — there is exactly one CSV schema in the project.
* Export uses the identical schema, listing only the words in the
  requested export region, ascending by address.

Rationale for 64-bit granularity: it is the AXI data width, so a record
maps 1:1 onto one `memory_t` word and no endianness reinterpretation ever
happens between Python, the CSV and the bus.

---

## 8 Top-level control interface

| Port | Dir | Type |
|---|---|---|
| `clk` | in | `std_ulogic` |
| `reset` | in | `std_ulogic` |
| `s_axi_lite_m2s` / `s2m` | in / out | `axi_lite_pkg` records |
| `m_axi_m2s` / `s2m` | out / in | `axi_pkg` records (AXI4, 64-bit) |
| `irq` | out | `std_ulogic` |

CSR map (32-bit registers, AXI4-Lite):

| Offset | Name | R/W | Bits |
|---|---|---|---|
| `0x00` | `CTRL` | RW | bit0 `START` (self-clearing), bit1 `SOFT_RESET` (self-clearing) |
| `0x04` | `PROGRAM_BASE_ADDR` | RW | program's first descriptor byte address |
| `0x08` | `STATUS` | RO/W1C | bit0 `BUSY`, bit1 `DONE` (sticky), bit2 `ERROR` (sticky), bits[7:4] `ERR_CODE`, bits[31:16] `ERR_PC_LOW` |
| `0x0C` | `IRQ_MASK` | RW | bit0 DONE, bit1 ERROR |
| `0x10` | `HW_INFO` | RO | [7:0] `PE_ROWS`, [15:8] `PE_COLS`, [23:16] `TILE_CHANNELS`, [31:24] `MAX_KERNEL_SIZE` |
| `0x14` | `HW_INFO2` | RO | [15:0] `ISA_VERSION` (`0x0200`), [31:16] `TENSOR_MEM_KIB` |
| `0x18` | `CMD_COUNT` | RO | descriptors retired |
| `0x1C` | `CYCLE_COUNT` | RO | cycles from `START` to `DONE` |
| `0x20` | `COMPUTE_CYCLES` | RO | cycles with an engine active |
| `0x24` | `STALL_CYCLES` | RO | cycles dispatched-but-blocked |
| `0x28` | `DDR_RD_BYTES` | RO | **all** AXI read bytes (incl. descriptors and weights) |
| `0x2C` | `DDR_WR_BYTES` | RO | all AXI write bytes |
| `0x30` | `TENSOR_LOAD_COUNT` | RO | `LOAD` commands retired |
| `0x34` | `TENSOR_STORE_COUNT` | RO | `STORE` commands retired |
| `0x38` | `WEIGHT_LOAD_BYTES` | RO | bytes fetched by `LOADW` + OT-loop refills |
| `0x3C` | `LOCAL_RD_BYTES` / `LOCAL_WR_BYTES` | RO | [15:0] KiB read / [31:16] KiB written in the scratchpad |

Counters are the mechanism the residency tests use to prove that
intermediates stayed local (§10). `DDR_WR_BYTES` is the decisive one: for a
multi-op local chain it must equal exactly the final `STORE` size.

---

## 9 Error policy — amendment to rev-1 decision D2

Rev 1 decision D2 said the hardware performs *no* legality checking
because "rejecting unsupported operations is the compiler's job". That
reasoning holds for **unsupported ops on a well-formed program**, and
`DWCONV2D` is still handled that way in spirit — but rev 2 adds a
scratchpad, and a malformed local address now **silently corrupts other
live tensors** instead of merely producing a wrong tensor. Silent
cross-tensor corruption is the single hardest class of bug to diagnose in
this system, so rev 2 **does** validate, in `cmd_proc`, before dispatch:

| `ERR_CODE` | Name | Condition |
|---|---|---|
| `0x0` | — | no error |
| `0x1` | `ERR_UNSUPPORTED_OP` | opcode not in §5.2, or `DWCONV2D` |
| `0x2` | `ERR_BAD_SPACE` | space tag `3`, or `LOCAL_WEIGHT` on a non-weight operand |
| `0x3` | `ERR_MISALIGNED` | any operand address not 8-byte aligned |
| `0x4` | `ERR_LOCAL_RANGE` | `LOCAL_TENSOR` access beyond `g_tensor_bytes` |
| `0x5` | `ERR_DDR_RANGE` | DDR access beyond `g_ddr_limit` |
| `0x6` | `ERR_BAD_RESERVED` | `W0[31:24] /= 0`, or `xfer_bytes /= 0` on a v1.2 opcode |
| `0x7` | `ERR_BAD_GEOMETRY` | zero/oversized dim, `kernel > g_max_kernel_size`, `in_width*ceil(Cin/8) > g_max_row_tile_words`, `stride = 0` |
| `0x8` | `ERR_AXI` | AXI `RRESP`/`BRESP` not OKAY |
| `0x9` | `ERR_TIMEOUT` | engine failed to complete within `g_watchdog_cycles` |

On any error: latch `ERR_CODE` and the offending PC, drop the command,
return to `IDLE`, set `STATUS.ERROR`, raise `irq` if unmasked. **The DUT
never hangs and never partially writes a validated-bad command's
destination.** `ERR_TIMEOUT` guarantees termination even if an engine
deadlocks, which is what makes the malformed-program tests non-hanging by
construction.

---

## 10 Residency policy, restated as testable invariants

| Invariant | How the test proves it |
|---|---|
| R1 | A compute command whose `space_dst = LOCAL_TENSOR` produces **zero** AXI write beats. `DDR_WR_BYTES` unchanged across it. |
| R2 | A chain of N compute commands with local operands produces AXI reads only for descriptors + weights, and AXI writes only for the closing `STORE`. |
| R3 | Local buffer contents are unchanged by an intervening unrelated command (read back via a later `STORE`). |
| R4 | A residual tensor written at command `i` and consumed at command `i+k` is never stored to DDR in between. |
| R5 | A forced spill (`STORE` to spill arena, later `LOAD` back) is bit-exact, and `TENSOR_STORE_COUNT` / `TENSOR_LOAD_COUNT` equal the program's explicit counts — no hidden traffic. |
| R6 | `weight_reuse=1` on a repeated conv adds **zero** to `WEIGHT_LOAD_BYTES`. |

---

## 11 Single-testbench architecture

Exactly one testbench: `test/tb_cnn_accel_top.vhd`. It is generic and
knows nothing about any network. Its entire per-test input is a set of
file paths passed as VUnit generics.

```
generics: g_mem_image_csv, g_result_csv, g_export_base, g_export_bytes,
          g_program_base, g_expect_error, g_timeout_cycles, ...
```

Flow:

1. `memory_t` created; `g_mem_image_csv` parsed and written word-by-word
   into it (`vunit_lib.memory_pkg.write_word`), permissions set
   read-and-write for the whole modelled range.
2. `bfm.axi_slave` bound to that `memory_t` serves the DUT's `m_axi`.
   This is the **only** path to memory the DUT has.
3. `reset` released; `bfm.axi_lite_master` writes `PROGRAM_BASE_ADDR`,
   then `CTRL.START`.
4. Poll `STATUS` until `DONE` or `ERROR` (or TB timeout -> fail).
5. Read every performance counter, write them to `<result>.counters.csv`.
6. Export `[g_export_base, g_export_base + g_export_bytes)` from the
   `memory_t` to `g_result_csv` in the §7 format.
7. `test_runner_cleanup`.

The testbench contains **no** convolution model, no expected values, no
tensor shapes, no opcode knowledge beyond the CSR map, and never touches a
DUT-internal signal. All numerical verification happens in Python
(`post_check`).

Feature variation comes exclusively from the generated CSV + generics, so
adding a test adds a Python function, never VHDL.

---

## 12 Known limitations of rev 2 (deliberate, documented)

1. `cmd_proc` dispatches **one command at a time** (no cross-command
   overlap). The banked scratchpad and the split request ports make
   overlap addable later without an ISA or port change; the OT loop
   already overlaps weight refill with nothing.
2. No `PSUM` in/out, so `Cin` tiling is still impossible — unchanged from
   the gap analysis (H7), and unnecessary once `g_max_row_tile_words` is
   raised (H2).
3. `DWCONV2D` remains unimplemented by design.
4. Scratchpad contents after `reset` are undefined by contract.
5. Only nearest-2x2 `UPSAMPLE` is implemented (the only mode YOLOv8n
   needs), not general `resize`.
6. `g_max_row_tile_words` stays at its current value in this phase; the
   H2 bump is a separate, independent change.
7. `dma_store` shares read channel `r0` with `engine_in_a` (§4), so a
   `STORE` cannot overlap a compute command's activation reads even once
   limitation 1 is lifted. Overlapping writeback with compute — the
   natural next optimisation after ping-pong loading — needs a third read
   channel. This costs nothing today because `cmd_proc` is sequential
   anyway, and it is a `cnn_accel_tensor_mem` port addition when wanted,
   not an ISA or memory-map change.
