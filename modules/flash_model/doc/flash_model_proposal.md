# flash_model — design proposal

Implements `flash_model_req.md`. This document records *why* the split is where
it is; the mechanical interface is in `flash_model_ffi_contract.md`.

## 1. Where the boundary sits

The VHDL component owns exactly two things Python cannot have: **pins** and
**simulation time**. Everything else is Python.

| VHDL | Python |
|---|---|
| Sample and drive `sck`/`cs_n`/`io` at the lane width it is told | Which lane width, how many dummy cycles, how many bytes |
| `wait for` a busy duration | How long each operation takes, and which commands are legal while it runs |
| Report `now` when asked | Derive write-in-progress from a stored deadline |
| Measure pin intervals, report violations, schedule output `after` a delay | The limit and delay values, per profile |
| Marshal `com` messages | The array, the command table, protection, status, SFDP, mode state |

The component contains **no flash knowledge**. It does not know that `0xEB` has
a mode byte, that `0x02` wraps within a page, or that programming is AND-only.
It shifts what the directive tells it to shift. That is the whole point: the
complexity lives where it can be unit-tested in milliseconds without a
simulator.

## 2. The directive

Per received or transmitted byte, Python answers one question: *what next?* The
answer is one packed integer describing one action — `receive`, `transmit` or
`ignore_rest` — with the lane width for that action, a count of dummy cycles to
insert **before** it, the byte to drive if transmitting, and two flags.

Three decisions inside that sentence were not obvious.

**Dummy cycles are a prefix, not a phase.** Modelling them as their own phase
forces the directive to carry two lane widths — one for the dummy cycles and one
for whatever follows — because the width changes *at* that boundary. `0x6B`
(address x1 → 8 dummy → data x4) is the case that exposes it. As a prefix on the
next action, one lane field suffices and one directive covers the transition.
`0x77` confirms the shape: there, dummy cycles precede a *receive*.

**Every field describes the same action.** An earlier draft had `lanes` mean
"for the next phase" while `byte_out` meant "drive this now" — two tenses in one
record, which is a durable source of off-by-one bugs. One tense throughout.

**`cs_assert` returns a directive too, not an acknowledgement.** The opening
lane width is not knowable by the component: in QPI mode the opcode itself is
x4, and after a `0xEB` with mode bits `M5:M4 = 10` the next transaction has no
opcode at all and starts at the address phase. The component cannot know either
of those; the model can. It also gives Python a way to say "ignore this entire
transaction".

**It is a packed scalar, not an array.** `python_pkg`'s `result_array`
allocates a fresh `integer_array_t` on every array-returning call. A
byte-granular hot path returning arrays would therefore allocate — and leak —
one array per byte shifted: four million for a 4 MiB read. A packed integer
allocates nothing. `decode_directive` turns it back into a record, so no bit
shifting appears at any call site.

## 3. Busy time is a deadline, not a flag

`cs_deassert` returns how long the operation takes and stores
`deadline = now + duration`. Write-in-progress is then *derived*
(`now < deadline`) whenever the model is asked, rather than latched.

The flag design that preceded it had the component set a busy flag, `wait for`,
then clear it. That has two defects. The `wait for` cannot live in the shift
engine — the component would be deaf to CS and SCK for the whole busy interval,
which is exactly when the testbench is polling status. Moving it to its own
process then creates a genuine cross-process delta race: the timer expiring and
the flag clearing land in the same simulation instant, with no defined order, so
a status byte shifted at that instant can report a stale bit.

A deadline has no flag to clear and therefore no race. The `wait for` survives
only so that a VHDL `flash_wait_until_ready` helper terminates; it is not the
source of truth, so a delta-order inversion against it is harmless. It also
leaves room for program/erase suspend, which needs elapsed time and which a
boolean cannot express.

`now` is passed on `cs_assert` always, and on `xfer` only when the previous
directive raised the `volatile` flag — so array reads pay nothing on the hot
path while status polling always sees a fresh timestamp.

## 4. Sparse storage

A dict of pages plus a run-length overlay for uniform fills. Absent data reads
`0xFF`. A 16 MiB device costs nothing until written; a 1 MiB `preload_fill` is
three integers across the boundary and no allocation at all.

This is why the array is **not** mirrored into VUnit's `memory_t`:
`memory_pkg.allocate()` is dense, and is already the reason `run.py` has to
raise `RLIMIT_STACK` to 512 MiB for the DDR model. A flash part is an order of
magnitude larger again.

Initialization is tiered so the bytes crossing the boundary stay bounded
regardless of image size: literals cross as bytes, uniform fills cross as a
length, and image files do not cross at all — Python opens them. Intel HEX and
S-record are themselves sparse formats, so a scattered image stays scattered.

## 5. The `python_execute` trap

Worth writing down, because it cost a real bug and the house guidance in
`shared/Vunit.md` §7 does not currently warn about it.

That section says calling `python_execute` more than once "is wasted work, not
an error, since it just re-executes the module". True for a single testbench
loading one bridge. Not true here. This VC loads its own bridge, by design, so
that a testbench never has to know the model is Python — which means two
instances execute the bridge file twice, and **re-executing a module resets its
globals**. The bridge's instance registry was defined at its module level, so
the second component's load wiped the first's registration and both were handed
id 1. Instance A read instance B's array, and three unrelated tests broke as
collateral.

The fix is the distinction between *executing* and *importing*. `python_execute`
re-runs a file; `import` resolves through `sys.modules`, which the simulator's
single embedded interpreter keeps for the whole simulation. So any state that
must outlive a reload belongs in an imported module — here,
`flash_model/registry.py` — and never at the bridge's own module level.

The rule for any future FFI-backed VC in this repo: **a bridge file that a VC
loads itself must be idempotent.** Treat its module body as something that can
run any number of times, and put every piece of surviving state behind an
import.

## 6. Known costs

One FFI call per byte on the read path. No allocation, but still a CPython call
— roughly 10 µs of host time for 20 ns of simulated time at x4/100 MHz.
Comfortable to about 64 KiB of bus traffic per test case; wrong for
image-sized reads. The extension, if it is ever needed, is chunked prefetch
behind an unchanged VHDL interface: the `n_bytes` field is already reserved,
Python picks the chunk length (and must pick 1 for any volatile byte, so
correctness stays on its side), and the component serves shifts from a local
cache.

The other cost is external: this component requires the VUnit fork carrying the
Python bridge, which is not the fork carrying headless `--wave`. Waveform
recording is unavailable until those are reconciled. See `requirements.txt`.
