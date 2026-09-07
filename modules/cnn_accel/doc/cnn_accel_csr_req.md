# cnn_accel_csr — requirement

## Responsibility

AXI4-Lite control/status register block, the host's only entry point:
program base address, start/abort control, done/error/IRQ status. Thin,
generic wrapper around `hdl-modules` `register_file.axi_lite_register_file`
(reused unmodified for the AXI4-Lite handshake/decode mechanics; this
module only defines the register map and the few bits of glue logic
around it — pulse generation, IRQ masking).

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_axi_lite_addr_width` | positive | register address width, propagated from `cnn_accel_top` |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk` | in | `std_ulogic` | |
| `reset` | in | `std_ulogic` | external reset only (see Clock/reset) |
| `s_axi_lite_m2s` | in | `axi_lite_pkg.axi_lite_m2s_t` | host access |
| `s_axi_lite_s2m` | out | `axi_lite_pkg.axi_lite_s2m_t` | host access |
| `program_base_addr` | out | `std_ulogic_vector(g_axi_addr_width-1 downto 0)` | to `cnn_accel_sequencer` |
| `start` | out | `std_ulogic` | one-cycle pulse, host-triggered program launch |
| `soft_reset_pulse` | out | `std_ulogic` | one-cycle pulse, drives `reset_internal` at `cnn_accel_top` |
| `seq_done` | in | `std_ulogic` | pulse from `cnn_accel_sequencer`: whole program halted normally |
| `seq_error` | in | `std_ulogic` | pulse from `cnn_accel_sequencer`: fatal error (bad opcode / AXI error response) |
| `irq` | out | `std_ulogic` | to top-level `irq` port |

## Register map (byte offsets, 32-bit registers)

| Offset | Name | R/W | Bits |
|---|---|---|---|
| `0x00` | `CTRL` | RW | bit0 `START` (self-clearing pulse), bit1 `ABORT`/`SOFT_RESET` (self-clearing pulse) |
| `0x04` | `PROGRAM_BASE_ADDR` | RW | full `g_axi_addr_width` bits, program's first instruction byte address |
| `0x08` | `STATUS` | RO | bit0 `BUSY`, bit1 `DONE` (sticky, write-1-to-clear), bit2 `ERROR` (sticky, write-1-to-clear) |
| `0x0C` | `IRQ_MASK` | RW | bit0 mask for `DONE`, bit1 mask for `ERROR` |
| `0x10` | `HW_INFO` | RO | bits[7:0] `PE_ROWS`, bits[15:8] `PE_COLS`, bits[23:16] `TILE_CHANNELS` -- elaborated array geometry (flow_status.md S3), so the host driver never hardcodes it |

## Clock/reset

Synchronous active-high `reset`, using the **unmodified external** `reset`
input — not `reset_internal` — for the register contents
(`PROGRAM_BASE_ADDR`, `IRQ_MASK`), per `doc/cnn_accel_arch.md` "Reset
policy" (an abort must not erase the address the host is trying to
re-launch). `STATUS`/`CTRL` pulse/sticky-bit logic and the `soft_reset_pulse`
generator itself may use either reset source since their steady-state
value after either kind of reset is the same (idle/clear).

<!-- functional-spec: hand-owned below this line -->

## Functional Description

- Writing `CTRL.START=1` while `STATUS.BUSY=0` pulses `start` for one
  cycle and sets `STATUS.BUSY=1`; `cnn_accel_sequencer` clears `BUSY` via
  `seq_done`/`seq_error`.
- Writing `CTRL.ABORT=1` at any time pulses `soft_reset_pulse` for one
  cycle (regardless of `BUSY`), which resets every datapath module in
  `modules/cnn_accel/` except this module's own configuration registers.
- `seq_done`/`seq_error` set `STATUS.DONE`/`STATUS.ERROR` (sticky) and
  clear `STATUS.BUSY`; `irq <= (STATUS.DONE and IRQ_MASK(0)) or
  (STATUS.ERROR and IRQ_MASK(1))`.
- Writes to `PROGRAM_BASE_ADDR` while `STATUS.BUSY=1` are accepted (AXI4-
  Lite `OKAY`) but have no effect until the next `START` (no mid-program
  address hot-swap).
