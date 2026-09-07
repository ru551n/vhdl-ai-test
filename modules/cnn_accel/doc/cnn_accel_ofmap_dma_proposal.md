# cnn_accel_ofmap_dma — vhdesign proposal

Input: `modules/cnn_accel/doc/cnn_accel_ofmap_dma_req.md`.
Related: `doc/cnn_accel_arch.md` ("Reset policy", "Interface record
policy"), `modules/cnn_accel/doc/cnn_accel_axi_read_dma_req.md` (sibling
module, same `dma_req`/`dma_done`/`resp_error` contract, same abort
rationale — used throughout as the precedent for anything this module's
own requirement leaves implicit), `hdl-modules/modules/dma_axi_write_simple`
and `hdl-modules/modules/ring_buffer` sources (read in full — see §3).

## 1. Requirements summary

Thin wrapper around `hdl-modules` `dma_axi_write_simple.dma_axi_write_simple`
(the un-wrapped, non-AXI-Lite entity) that adapts its native ring-buffer
register interface to this IP's one-shot `dma_req_m2s_t`/`s2m_t`
(`addr`+`length`) control plane. Accepts one AXI4-Stream of int8 output
pixels per request (`s_stream_m2s`/`s2m`), writes exactly `length` bytes
starting at `addr` via AXI4 write bursts (`m_axi_aw`/`w`/`b`, split
channel slices, no arbitration — single writer straight to the top-level
`m_axi`), and reports `dma_done` (pulse, full length written) /
`resp_error` (pulse, non-`OKAY` `BRESP` seen) back to `cnn_accel_layer_ctrl`.
Synchronous active-high `reset` (`= reset_internal`), same abort rationale
as `cnn_accel_axi_read_dma`: an in-flight write must be abortable without
leaving a stale outstanding-beat count or a stuck `req_s2m.ready = '0'`.

## 2. Interface (as given by the requirement)

| Port | Dir | Type |
|---|---|---|
| `clk` | in | `std_ulogic` |
| `reset` | in | `std_ulogic := '0'` |
| `req_m2s` | in | `cnn_accel_pkg.dma_req_m2s_t` |
| `req_s2m` | out | `cnn_accel_pkg.dma_req_s2m_t` |
| `dma_done` | out | `std_ulogic` |
| `resp_error` | out | `std_ulogic` |
| `s_stream_m2s` | in | `axi_stream_pkg.axi_stream_m2s_t` |
| `s_stream_s2m` | out | `axi_stream_pkg.axi_stream_s2m_t` |
| `m_axi_aw_m2s` | out | `axi_pkg.axi_m2s_a_t` |
| `m_axi_aw_s2m` | in | `axi_pkg.axi_s2m_a_t` |
| `m_axi_w_m2s` | out | `axi_pkg.axi_m2s_w_t` |
| `m_axi_w_s2m` | in | `axi_pkg.axi_s2m_w_t` |
| `m_axi_b_m2s` | out | `axi_pkg.axi_m2s_b_t` |
| `m_axi_b_s2m` | in | `axi_pkg.axi_s2m_b_t` |

Generics: `g_axi_addr_width : positive`, `g_axi_data_width : positive`
(constrained internally to `axi_pkg.axi_data_width_t`, i.e. must be one of
the AXI-legal data widths, 8/16/32/64/...). No other generics are exposed
— see §3.4 for why the wrapped core's `stream_data_width` is *not* an
independent generic here.

`req_m2s`/`req_s2m` are two separate signals (the requirement's port table
row groups them for brevity, as does `cnn_accel_axi_read_dma_req.md`'s for
some rows, but that module's own table gives them separate rows — same
resolution applied here). `dma_req_t.addr`/`.length` are both
`unsigned(31 downto 0)` per `cnn_accel_pkg.vhd`.

## 3. Design decisions (resolving requirement ambiguities / wrapped-module fit)

The requirement text is deliberately terse ("forward them to the wrapped
`dma_axi_write_simple` instance's control interface"); `dma_axi_write_simple`
itself, however, is not a simple addr+length one-shot engine — it is a
*continuously-running ring buffer* with host-managed read/write pointers,
designed for a CPU that keeps consuming a circular DMA buffer forever. All
four of the below decisions exist to make that shape fit the one-shot
`dma_req` contract *without modifying the wrapped module*. This section
records what was actually read in the wrapped module's RTL (not just its
`.rst`/header prose), because two of the four decisions below hinge on
behaviour the header text does not fully spell out.

### 3.1 Instantiate the plain entity, not `dma_axi_write_simple_axi_lite`

`dma_axi_write_simple`'s `regs_up`/`regs_down` ports
(`dma_axi_write_simple_regs_down_t`/`_up_t`, from
`dma_axi_write_simple_register_record_pkg`) are plain, un-latched record
ports at this level — the "software must not change `buffer_start_address`
once enabled" caveats in `regs_dma_axi_write_simple.toml` describe the
*AXI-Lite register file's* usage contract (`dma_axi_write_simple_axi_lite`,
not instantiated here), not a hardware interlock inside the plain entity
itself. Using the plain entity and driving `regs_down` directly from this
wrapper's own FSM (no AXI-Lite protocol, no register-file underneath) is
what makes a per-request reconfiguration of `buffer_start_address`/
`buffer_end_address`/`buffer_read_address` possible at all; this is the
crux of the whole adaptation and is *not* optional.

### 3.2 Degenerate one-shot mapping onto the ring buffer

Confirmed from `ring_buffer_write_simple.vhd`'s `main` process: on the
`enable` rising edge (`enable = '1' and enable_p1 = '0'`), the core
unconditionally reinitializes `buffer_written_index`/`segment_index` to
the *current* `buffer_start_address`. This means a "single-shot buffer"
can be re-issued correctly, request after request, purely by toggling
`enable` low then high with new addresses in between — no ring-buffer
wraparound tracking is needed as long as each request's region is treated
as fully disposable (never revisited). Per request:

- `buffer_start_address <= addr`
- `buffer_end_address <= addr + length`
- `buffer_read_address <= addr + length` (the *whole* region is declared
  "already free", i.e. software has notionally already consumed
  everything up to the end before the module even starts — there is no
  real second consumer, so there is nothing to actually hold back for)
- `enable`: pulsed from `'0'` to `'1'` for the duration of this request
  only (see §3.3 for exactly how "duration" is bounded), then dropped
  back to `'0'` before the next request's `enable` rising edge, so the
  reinitialization above fires exactly once per request.

This is the "single-shot buffer" mapping anticipated in this session's
opening analysis, now confirmed against the actual ring-buffer RTL rather
than assumed from the register `.toml` prose alone.

### 3.3 `packet_length_beats` generic: forced to the minimum legal value

`dma_axi_write_simple` requires `packet_length_beats` (a *compile-time*
generic — the header is explicit that runtime partial-packet flushing is
not supported) and asserts `packet_length_bytes mod axi_data_width_bytes
= 0` plus power-of-two constraints. This wrapper sets
`stream_data_width => g_axi_data_width` (§3.4) and
`packet_length_beats => 1`, which resolves to exactly one packet = one
native AXI beat (`packet_length_axi_beats = 1`), landing on the module's
own "optimized implementation for single-beat packets" generate branch
(no persistent multi-beat FSM inside the wrapped core at all — each
accepted stream beat becomes its own `AW`+`W` burst of length 1).

Consequences, both documented as the deliberate trade-off of staying
thin rather than as a defect:

- **Alignment requirement pushed onto the caller**: `addr` and `length`
  must both be multiples of `g_axi_data_width/8` bytes. This is not an
  *extra* constraint invented by this wrapper — `dma_axi_write_simple`
  always drives a full-width `WSTRB` (`to_strb(axi_data_width)`, no narrow
  writes) and has no partial-packet flush at any `packet_length_beats`
  value, so byte-granular `length` is unsupported by the wrapped core
  itself, for *any* packet size. Choosing the minimum packet size (one
  AXI beat) means this is the *only* alignment constraint — the smallest
  one achievable with this core, not an additional one. Recorded as an
  **open question** (§7) for `cnn_accel_layer_ctrl`/the architect: is the
  per-tile ofmap byte count always known to be a multiple of
  `g_axi_data_width/8`?
- **Burst efficiency**: every beat is its own single-beat `AW` burst —
  far below "AXI bursts of the maximum length possible" (the wrapped
  module's own stated design goal in the general case). A larger
  `packet_length_beats` would coalesce multiple stream beats into one
  real burst, but at the cost of a coarser, harder-to-guarantee alignment
  requirement (a whole packet's worth of bytes, not one AXI beat) with
  the same "no partial packet" failure mode if violated (bytes silently
  stall forever inside the core, `dma_done` never fires). Recorded as a
  second **open question** (§7): burst efficiency is sacrificed here for
  robustness against any `length` that merely happens to be
  beat-aligned; revisit with a real `packet_length_beats > 1` (and a
  matching alignment contract on the caller) if DDR write bandwidth on
  the ofmap path turns out to be the bottleneck.
- Since `packet_length_axi_beats = 1`, "one packet accepted" and "one AXI
  beat's `BRESP` received" are the same event — completion tracking
  (§3.5) can watch the `B` channel directly and never needs to touch
  `regs_up`/`interrupt` at all.

### 3.4 `stream_data_width` generic tied to `g_axi_data_width` (no width conversion)

`dma_axi_write_simple` instantiates `common.width_conversion` internally
whenever `stream_data_width /= axi_data_width`. That block's internal
beat-accumulation counter has **no reset port** (see §4) — if a request
is aborted mid-packet-accumulation, the next request's very first
accumulated AXI beat would silently be corrupted by the previous
request's leftover partial bytes, with no way for this wrapper to flush
or observe that internal counter from the outside. This wrapper avoids
the failure mode entirely by construction: it requires the producer's
`s_stream_m2s.data` to already be `g_axi_data_width` bits of valid
payload per beat (only the low `g_axi_data_width` bits of the
fixed-width `axi_stream_m2s_t.data` field are used), so `stream_data_width
= axi_data_width` is passed to the wrapped core and its
`width_conversion` generate branch is never elaborated. No generic is
exposed for a narrower stream width — see the third **open question**
(§7): does `cnn_accel_bias_requant`/`cnn_accel_pool`'s converged output
mux actually deliver `g_axi_data_width` bits/beat at the point it reaches
this module? If not (e.g. it delivers one int8 lane, or `g_pe_rows`
lanes, at some width narrower than the AXI bus), this generic tie is
currently the load-bearing assumption that keeps this wrapper thin and
correct, and needs either an architect decision on the mux's output
width or a (non-thin, out-of-scope-for-this-module) width-conversion
stage added ahead of `s_stream_m2s`.

### 3.5 Completion/error tracking: tap `m_axi_b_*` directly, bypass `regs_up`/`interrupt`

Because one packet = one `AW`/`W`/`B` transaction (§3.3),
`dma_done`/`resp_error` are derived directly from counting `B`-channel
handshakes this wrapper already passes through to `m_axi_b_m2s`/`s2m`
(`m_axi_b_m2s.ready = '1'` always, matching the wrapped core's own
`axi_write_m2s.b.ready <= '1'` — see §5), rather than by reading
`regs_up.buffer_written_address` (which wraps at `buffer_end_address`,
never assumes that value, and would need modular-arithmetic handling for
no benefit) or `regs_up.interrupt_status.write_done` (a
read-clear-by-write-1 status bit with no benefit over directly observing
the same event on the port this wrapper already owns). `regs_up` and
`interrupt` are left unconnected (`open`); `regs_down.interrupt_status`/
`.interrupt_mask` are tied to their `_init` values (all-zero — never
clears anything, never masks anything, both irrelevant since nothing
reads `regs_up`/`interrupt` here).

Per-request beat accounting (all in the wrapper's own resettable state):

- `expected_beats_q`: latched at request-accept time = `length` shifted
  right by `log2(g_axi_data_width/8)` bits (§3.3's alignment means this
  is exact, not truncating).
- `aw_issued_q`: counts `m_axi_aw_m2s.valid and m_axi_aw_s2m.ready`
  handshakes since this request's `enable` rising edge. Drives
  `regs_down.config.enable` combinationally:
  `enable <= '1' when state_q = active and aw_issued_q < expected_beats_q
  else '0'` — this is what stops the ring buffer from ever requesting a
  segment *beyond* this request's length (recall `buffer_read_address`
  was set equal to `buffer_end_address` in §3.2, so the ring buffer alone
  has no notion of "this region is now full" — it would otherwise loop
  back to `buffer_start_address` and keep issuing more segments
  indefinitely). Because `enable` is combinational in `aw_issued_q`
  (itself registered), the drop to `'0'` takes effect for the ring
  buffer's very next `idle`-state evaluation, one cycle after the
  request's last `AW` handshake — no race against a spurious extra
  segment.
- `bresp_acked_q`: counts `B`-channel handshakes since the same rising
  edge; separate from `aw_issued_q` because `dma_done` must mean "fully
  *written*" (response received), not merely "fully issued".
  `bresp_acked_q + 1 = expected_beats_q` on a `B` handshake pulses
  `dma_done` and returns to `idle`.
- `error_latched_q`: set on any `B` handshake with `resp /= OKAY`; not
  pulsed immediately (mirrors `cnn_accel_axi_read_dma`'s own documented
  policy — "still drain ... but latch and pulse `resp_error` once the
  request completes"). `resp_error <= error_latched_q or (this cycle's
  resp /= OKAY)` is asserted on the same cycle as the completing
  `dma_done` pulse, not before.

## 4. Clock/reset — the wrapped core has no reset port at all

Verified by grep across `dma_axi_write_simple.vhd`,
`ring_buffer_write_simple.vhd`, and the `_axi_lite` wrapper: **zero**
matches for "reset". This is `hdl-modules`' project-wide resetless-by-
default convention (`doc/cnn_accel_arch.md`'s "Reset policy" section
explicitly calls this project default out before describing why
`modules/cnn_accel/` overrides it for its own modules) — it is not an
oversight, and it cannot be worked around by "fixing" the wrapped module
(that would violate the "reused unmodified" mandate in this module's own
Responsibility text, and the `AGENTS.md` no-fork rule).

This wrapper resolves the conflict the same way the architecture doc's
own reset-policy rationale implies for any resetless reused primitive: it
does not attempt to reset the wrapped core's internals directly (there is
no port to reset), and instead:

1. Owns 100% of the state that `req_s2m.ready`/`dma_done`/`resp_error`
   depend on (`state_q`, `expected_beats_q`, `aw_issued_q`,
   `bresp_acked_q`, `error_latched_q`, the latched `addr`/`length`) in
   registers that `reset` *does* synchronously clear on this side of the
   boundary.
2. Never asks the wrapped core to abandon an `AW`/`W` transaction that is
   already asserted on the bus — that is not a `dma_axi_write_simple`
   limitation, it is a basic AXI4 rule (a master may not withdraw
   `AWVALID`/`WVALID` before the corresponding handshake); "abort" for
   any AXI4 write master can only mean *stop issuing new bursts*, not
   *retract an in-flight one*.
3. Tracks physical in-flight `AW`-accepted-but-`B`-not-yet-seen
   transactions in one small counter, `outstanding_q`
   (`unsigned(7 downto 0)`, sized generously for realistic interconnect
   pipelining depth), that **`reset` deliberately does not clear** — by
   design, not omission. Clearing it on `reset` would make the wrapper
   falsely believe the bus was already quiet, exactly the "stale
   outstanding-beat count" failure mode the requirement calls out by
   name. `outstanding_q` increments on an `AW` handshake, decrements on a
   `B` handshake, unconditionally, forever — it is the one piece of
   state in this module that must survive `reset` to remain physically
   accurate.
4. On `reset = '1'`: `req_s2m.ready`, `enable`(via `state_q`, see §3.5),
   `aw_issued_q`, `bresp_acked_q`, `error_latched_q`, `dma_done`,
   `resp_error` all clear/deassert synchronously. `state_q` goes to
   `s_idle` if `outstanding_q`'s *next* value (computed the same cycle,
   accounting for any `AW`/`B` handshake happening on the reset cycle
   itself) is already `0`, otherwise to a dedicated `s_drain` state.
5. `s_drain`: `req_s2m.ready` held `'0'`, `enable` held `'0'` (so the
   ring buffer cannot issue any new segment — the only thing that could
   still change `outstanding_q` is a `B` response draining an
   already-issued transaction). Once `outstanding_q` reaches `0`
   (guaranteed in bounded time: no new `AW`s can be issued while
   `enable = '0'`, and the wrapped core ties its own `axi_write_m2s.b.
   ready` high internally, so every already-issued transaction's `B`
   response is accepted the moment the slave provides it — per AXI4
   rule 15 a compliant slave always eventually does), `state_q` returns
   to `s_idle` and `req_s2m.ready` goes back high. This is "not stuck at
   `'0'`" (the requirement's actual wording) without requiring the
   impossible "instantaneous" reading of that phrase.
6. Normal (non-aborted) completion reaches `outstanding_q = 0` at
   exactly the same cycle `bresp_acked_q` reaches `expected_beats_q`
   (§3.3's `aw_issued_q`-gated `enable` guarantees `aw_issued_q` for a
   request never exceeds `expected_beats_q`, and in-order same-non-ID
   `B` responses per rule 13's single-ID simplification mean the last
   `B` for this request's last `AW` is also the point `outstanding_q`
   bottoms out) — so `s_drain` is only ever entered on an aborted
   (mid-request) `reset`, never on the normal completion path.

`s_stream_s2m.ready` needs no separate reset-time gating at all: it is
wired straight through from the wrapped core's own `stream_ready` (§5),
which the ring buffer itself already forces low whenever `enable =
'0'` (no segment ever becomes valid while disabled) — i.e. the wrapped
core's own enable-gated behaviour already produces exactly the
backpressure this wrapper needs during `s_idle`/`s_drain`, with no extra
logic required.

## 5. Architecture and dataflow

```
                     +-----------------------------------------------+
 req_m2s/s2m ------->|  control FSM (s_idle / s_active / s_drain)    |
 dma_done    <-------|  - latches addr/length                       |
 resp_error  <-------|  - expected_beats_q / aw_issued_q /           |
                     |    bresp_acked_q / error_latched_q            |
                     |  - outstanding_q (reset-surviving)            |
                     +-----------------+-------------------+---------+
                                       | regs_down          | (combinational
                                       | (buffer_start/end/ |  enable)
                                       |  read_address,     |
                                       |  config.enable)    |
                                       v                    |
 s_stream_m2s/s2m <---------------------------------------->|
   (direct passthrough: stream_valid/data <= s_stream_m2s,  |
    s_stream_s2m.ready <= stream_ready)                     |
                                       v                    |
                     +-----------------------------------------------+
                     |   dma_axi_write_simple (plain entity,          |
                     |   packet_length_beats = 1, stream_data_width   |
                     |   = axi_data_width = g_axi_data_width)         |
                     |   regs_up, interrupt: open (unused, §3.5)      |
                     +-----------------+-------------------+---------+
                                       | axi_write_m2s/s2m (aw/w/b)
                                       v
                     split into m_axi_aw_m2s/s2m, m_axi_w_m2s/s2m,
                     m_axi_b_m2s/s2m (pure field fan-out/fan-in,
                     no logic) <----- also snooped for aw_issued_q/
                                      bresp_acked_q/outstanding_q/
                                      error_latched_q (§3.5, §4)
```

`axi_write_m2s.aw/.w/.b` <-> `m_axi_aw_m2s`/`m_axi_w_m2s`/`m_axi_b_m2s`
(and the `_s2m` counterparts) is pure record field assignment; no logic
is added on this path — the "thin wrapper" constraint is satisfied for
the entire AXI4 write master boundary, not just the control plane.

### State machine

```
s_idle   : req_s2m.ready = '1'. On req_m2s.valid: latch addr/length,
           compute expected_beats_q. If expected_beats_q = 0,
           -> s_zero_len (see below). Else -> s_active,
           aw_issued_q/bresp_acked_q/error_latched_q <= 0.
s_zero_len: one-cycle state; pulses dma_done, -> s_idle. (A length=0
           request is legal per the requirement's addr+length shape and
           trivially "fully written" with zero bursts.)
s_active : enable driven combinationally per §3.5. On a B handshake:
           bresp_acked_q += 1; latch error_latched_q on resp /= OKAY;
           if bresp_acked_q + 1 = expected_beats_q, pulse dma_done
           (and resp_error if any error was seen, including this one),
           -> s_idle.
s_drain  : entered only from a reset with outstanding_q /= 0 (§4).
           req_s2m.ready = '0', enable = '0'. -> s_idle once
           outstanding_q = 0.
```

`outstanding_q` (§4) is updated identically in every state, including
during `reset` — it is not part of the `state_q` case statement's
per-state logic, it is unconditional.

## 6. Numeric types and widths

- `dma_req_t.addr`/`.length`: `unsigned(31 downto 0)` (from
  `cnn_accel_pkg.vhd`, not redefined here).
- `expected_beats_q`, `aw_issued_q`, `bresp_acked_q`: `unsigned(31 downto
  0)` — sized to match `length`'s width rather than a tightly-computed
  minimum; this is a `g_axi_data_width`-independent, always-safe choice
  appropriate for a first thin implementation (see `shared/
  DesignPatterns.md`/reset-minimization guidance on not over-optimizing
  register widths ahead of a real resource-usage measurement).
- `outstanding_q`: `unsigned(7 downto 0)` — a deliberately generous fixed
  bound on realistic AXI interconnect/DDR-controller outstanding-
  transaction pipelining depth (not generic-driven; flagged as an
  assumption, not a hard architectural limit, in §7).
- All address/length arithmetic (`addr + length`, the `shift_right` used
  to derive `expected_beats_q` from `length`) uses `ieee.numeric_std`
  `unsigned`, never `std_logic_vector` arithmetic, per `AGENTS.md`.
- `regs_down`'s `register_t` fields (32-bit `std_ulogic_vector`, from
  `register_file.register_file_pkg`) are populated via
  `std_ulogic_vector(<32-bit unsigned>)` direct conversion (both exactly
  32 bits — `register_width = 32` in `register_file_pkg.vhd` — no
  resize needed, but written as `resize(..., register_t'length)` anyway
  for robustness against a future `register_width` change).

## 7. Open questions

1. **Ofmap byte-length alignment** (§3.3): is `cnn_accel_layer_ctrl`
   guaranteed to only ever issue `dma_req`s whose `length` (and `addr`)
   are exact multiples of `g_axi_data_width/8` bytes? This wrapper
   requires it (inherent to `dma_axi_write_simple`, not an extra
   constraint invented here — see §3.3) and has no way to detect or
   report a violation (an unaligned tail would simply never complete —
   `dma_done` would never fire, silently hanging the layer).

   **Update, decision S6** (see `doc/cnn_accel_arch.md`, "Off-chip
   activation layout"): the strided-ofmap-write conflict this item was
   waiting on is resolved — the ofmap layout is channel-tiled planes
   `[C/8][H][W][8]`, so every `cnn_accel_layer_ctrl` write-back is one
   contiguous plane and `dma_req_t` is unchanged. That also *mostly*
   answers this alignment question, since both the plane length and every
   plane's start offset are multiples of `T = 8` bytes:
   ```
   addr   = ofmap_addr + plane_idx * (out_width * out_height * 8)
   length =                          out_width * out_height * 8
   ```
   So the request is inherently 8-byte aligned. At `g_axi_data_width = 64`
   (8 bytes/beat) the requirement is therefore satisfied unconditionally.
   At a wider bus it is *not* automatic: e.g. at 128 bits (16 bytes/beat)
   `out_width * out_height` must additionally be even, and the base
   `ofmap_addr` the host programs must be 16-byte aligned.

   **RESOLVED by decision D1 (2026-09-07): bound the bus width.** Of the
   three candidate enforcements — (i) elaboration assert capping
   `g_axi_data_width`, (ii) a runtime check in `cnn_accel_layer_ctrl`
   raising `layer_error`, (iii) a documented host/compiler contract with no
   hardware check — (i) was ratified. `cnn_accel_constants.py` now declares
   `ACTIVATION_PLANE_CHANNELS = 8` (T, a memory-layout constant kept
   deliberately distinct from the datapath's `TILE_CHANNELS`) and
   `MAX_AXI_DATA_WIDTH = 8 * ACTIVATION_PLANE_CHANNELS`; both are
   propagated by `hdl-registers` and asserted concurrently, at
   `severity failure`, in **`cnn_accel_ofmap_dma`** and
   **`cnn_accel_axi_read_dma`** (the latter because S6 makes its ifmap
   instance plane-granular too).

   Rationale for (i) over (ii)/(iii): the guarantee becomes independent of
   runtime layer geometry, costs no logic, needs no new error cause, and
   cannot be violated by a host that programs a legal-looking descriptor.
   `out_width`/`out_height` are runtime descriptor fields, so the general
   alignment predicate is *not* statically checkable — only the bus-width
   bound is. Accepted cost: 128-bit AXI is forbidden, capping ofmap write
   and ifmap read burst bandwidth. Revisit only if a real measurement shows
   the 64-bit bus is the bottleneck, and note that lifting the bound means
   adopting (ii), not simply widening the generic.
2. **Burst efficiency vs. robustness trade-off** (§3.3): this proposal
   deliberately picks the smallest legal `packet_length_beats` (single-
   AXI-beat bursts) to make the "no partial packet" limitation
   disappear into the ordinary AXI beat-width alignment every AXI write
   master already needs. If ofmap write bandwidth turns out to be a
   real bottleneck, a larger `packet_length_beats` is possible but
   trades away robustness against any length that isn't a multiple of
   the (now much larger) packet size — needs a real measurement before
   revisiting, not a guess.
3. **Producer stream width** (§3.4): does the `cnn_accel_bias_requant`/
   `cnn_accel_pool` converged output mux actually deliver
   `g_axi_data_width` bits of payload per beat at the point it reaches
   `s_stream_m2s`? This wrapper's avoidance of the wrapped core's
   internal (reset-less) `width_conversion` block depends on it. If the
   mux's native beat width is narrower (e.g. one int8 lane, or
   `g_pe_rows` lanes at some width below the AXI bus), this needs an
   architect decision (widen the mux's output, or accept a
   non-thin width-matching stage ahead of this module — out of scope
   for `cnn_accel_ofmap_dma` itself either way).
4. **`outstanding_q` width** (§6): fixed at 8 bits (255 outstanding
   transactions) as a generous-but-unverified bound. Not generic-driven.
   Revisit once the top-level AXI interconnect/DDR controller's actual
   outstanding-transaction capability is known; overflow would wrap
   `outstanding_q` silently (undercounting drain requirements is the
   unsafe direction — this should be widened rather than narrowed if
   ever in doubt).

## 8. Verification plan

Bench: `tb_cnn_accel_ofmap_dma.vhd`, `bfm.axi_write_slave` terminating
`m_axi_aw`/`w`/`b` against the VUnit memory model. Written data is checked
byte-exactly with `set_expected_word` + `check_expected_was_written`;
directed AXI backpressure and held-back `BRESP`s come from switching the
slave's per-channel stall probabilities between 0.0 and 1.0 at runtime.
VUnit's slave always answers `BRESP=OKAY`, so the `resp_error` case uses a
passive wire-level override of the `resp` field between BFM and DUT for one
chosen B beat. The producer stream is driven by a small `ready`-honoring
procedure because several cases interleave cycle-exact checks between
individual beats.

- **Single request, single beat**: `length = g_axi_data_width/8` bytes,
  one AXI beat, verify `AW`/`W` issued once with the right `addr`, one
  `BRESP = OKAY` accepted, `dma_done` pulses exactly once, `resp_error`
  never asserted, `req_s2m.ready` returns high the next cycle.
- **Single request, multiple beats**: `length` a multiple of several
  beats; verify beat count, address increment per beat (each beat is its
  own `AW`, address = `addr + n*bytes_per_beat`), `dma_done` timing
  (exactly on the last `BRESP`, not before).
- **Back-to-back requests**: second `req_m2s.valid` presented immediately
  after the first `dma_done`; verify the second request's `AW` addresses
  start fresh at its own `addr` (ring-buffer reinit per §3.2 actually
  takes effect) and its own beat count is independent of the first.
- **`resp_error` (`BRESP = SLVERR`/`DECERR`)**: verify `dma_done` still
  pulses (full length was *attempted*/drained) and `resp_error` pulses on
  the same cycle; verify an error partway through a multi-beat request is
  latched and still reported at completion, not immediately.
- **Stream backpressure**: `s_stream_m2s.valid` deasserted mid-request
  (stall); verify `m_axi_aw`/`w` correctly stall too (no burst started
  without data ready) and the request still completes once the stream
  resumes.
- **AXI backpressure**: `m_axi_aw_s2m.ready`/`m_axi_w_s2m.ready` held low
  for several cycles; verify no protocol violation (`AWVALID` held
  stable while not accepted) and eventual completion.
- **Abort mid-request (the reset/`s_drain` path, §4)**: assert `reset`
  while `outstanding_q /= 0` (at least one `AW` accepted, `BRESP` not
  yet returned); verify `req_s2m.ready` drops immediately and returns
  high only after the pending `BRESP`(s) are observed on `m_axi_b_s2m`,
  never before; verify a fresh request issued once `req_s2m.ready`
  returns high behaves identically to any other first request (no
  corruption from the aborted one — in particular check the first
  beat's address and data are exactly the new request's, not
  contaminated by anything left over).
- **Abort with zero outstanding**: assert `reset` while `state_q =
  s_idle` or exactly at a request boundary; verify `req_s2m.ready`
  returns high the very next cycle (no spurious `s_drain` detour).
- **`length = 0`**: verify `dma_done` pulses with no `AW`/`W` activity
  at all, `req_s2m.ready` returns high one cycle later.

## Implementation Notes (vhfill)

(empty — filled in during/after RTL implementation)
