# flash_model — requirements

A QSPI NOR flash **verification component** for VUnit testbenches. It presents a
JEDEC-conformant flash device on a QSPI bus so that a controller DUT can be
verified against something that behaves like a real part — including the parts
of "real" that are easy to get wrong: NOR's AND-only programming, page-program
wraparound, write-enable latching, erase granularity, block protection, and
program/erase busy time.

The entire device model is Python. The VHDL component is a pin-level shift
engine that owns only what Python cannot: driving wires and knowing what time it
is.

## R1 — Interface

- **R1.1** The component shall present a QSPI slave interface: `sck`, `cs_n` and
  a bidirectional 4-bit `io` bus, with x1, x2 and x4 lane widths.
- **R1.2** It shall be instantiated and controlled exactly like a standard VUnit
  verification component: a handle record passed as a generic, and procedures
  that send `com` messages to the handle's actor. A testbench shall never drive
  the flash pins directly and shall never mention Python.
- **R1.3** It shall implement the `sync` VCI (`as_sync`, `wait_until_idle`).
- **R1.4** Multiple instances shall coexist in one testbench without shared
  state.

## R2 — Command set

- **R2.1** It shall decode the JEDEC baseline command set: read (`0x03`), fast
  read (`0x0B`), dual/quad output and dual/quad I/O reads (`0x3B`, `0x6B`,
  `0xBB`, `0xEB`), page program (`0x02`, `0x32`), sector and block erase
  (`0x20`, `0x52`, `0xD8`), chip erase (`0xC7`/`0x60`), write enable/disable
  (`0x06`, `0x04`), status register read/write (`0x05`, `0x35`, `0x15`, `0x01`),
  JEDEC ID (`0x9F`) and SFDP (`0x5A`).
- **R2.2** It shall support **both** 3-byte and 4-byte addressing, including the
  mode-switch commands (`0xB7`, `0xE9`) and the dedicated 4-byte opcodes
  (`0x13`, `0x0C`, `0x12`, `0xDC`).
- **R2.3** It shall support QPI mode (`0x38` enter, `0xFF` exit), in which the
  opcode itself is transferred at x4.
- **R2.4** It shall support continuous-read / XIP: when a `0xEB` carries mode
  bits `M5:M4 = 10`, the **next** transaction has no opcode and begins at the
  address phase.
- **R2.5** An unrecognised or currently-illegal command shall be ignored (clocks
  consumed, nothing driven, nothing changed) and counted, not treated as an
  error — that is what a real part does.

## R3 — Array semantics

- **R3.1** Programming shall be AND-only: a programmed byte becomes
  `old AND new`. A bit can only go 1 → 0 outside an erase.
- **R3.2** Erase shall set every byte in the erased unit to `0xFF`, at sector,
  32 KiB block, 64 KiB block and whole-chip granularity.
- **R3.3** A page program shall latch at most one page and wrap **to the start
  of the same page**, never into the next page.
- **R3.4** A program or erase shall be ignored unless the write-enable latch is
  set, and shall clear it on completion.
- **R3.5** Storage shall be **sparse**: memory proportional to what has actually
  been written, not to the device size. Unwritten locations read `0xFF`.

## R4 — Initialization and inspection

- **R4.1** The array shall be initializable sparsely from VHDL, at three scales:
  scattered literal regions; a large uniform fill that is O(1) in its length;
  and an image file that Python reads directly, so only its name crosses the
  language boundary.
- **R4.2** Image loading shall accept Intel HEX, Motorola S-record, raw binary
  and JSON, preserving the sparseness of the natively sparse formats.
- **R4.3** Contents shall be readable back into VHDL, and checkable in Python so
  a mismatch reports the address and both values.
- **R4.4** The set of regions the DUT has written shall be reportable, as a
  scoreboard for "did it write only where it should".
- **R4.5** Initialization shall bypass protection and the write-enable latch: it
  is test setup, not a device operation.

## R5 — Timing

- **R5.1** Program, erase, status-write, reset and power-down operations shall
  hold write-in-progress for a configurable, per-operation duration.
- **R5.2** A single switch shall collapse every such duration to zero, so a test
  that does not care about timing pays nothing for it.
- **R5.3** Individual durations shall be overridable per test.
- **R5.4** While write-in-progress, every command except a status read shall be
  ignored, and status polling shall continue to work — the component shall not
  be deaf to the bus while it is busy.
- **R5.5** Pin-level timing shall be checked against the device profile: SCK
  period and high/low time, CS setup and hold, and CS deselect between commands.
  Violations shall be reported through the component's own checker so a negative
  test can mock it.
- **R5.6** Clock-to-output and output-disable times shall be applied as **output
  delays**, not checked — they describe what the component drives, not what the
  DUT does.

## R6 — Failure reporting

- **R6.1** Model-level failures (content mismatch, a malformed request, an
  internal inconsistency) shall raise in Python, so the failure arrives as a
  VUnit FAILURE carrying a full traceback.
- **R6.2** Pin-level timing violations, which VHDL detects, shall be reported
  through the component's `checker_t`.
- **R6.3** The VHDL and Python halves shall agree on their shared data layout by
  construction: a version handshake at model-creation time shall fail
  immediately on drift, rather than surfacing as a wrong byte later.

## R7 — Portability

- **R7.1** Everything shall compile and pass on **both** GHDL and NVC, which is
  this project's CI matrix.
- **R7.2** The component shall be simulation-only and never enter synthesis.

## Non-goals (this iteration)

Program/erase suspend and resume; OTP/security registers; individual block lock
commands (`0x36`/`0x39`); DTR/octal modes; multi-die stacked parts. The
directive layout and the timing table leave room for each.
