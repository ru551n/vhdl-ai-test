# cnn_accel_sequencer — requirement

## Responsibility

Program-counter and instruction decode: on `start`, issues a
`dma_req` for `g_instr_word_bytes` bytes at `program_base_addr` to its own
`cnn_accel_axi_read_dma` instance, assembles the returned AXI4-Stream
beats into one 64-byte descriptor, decodes it into a `layer_desc_t`
(`cnn_accel_pkg`), and hands it to `cnn_accel_layer_ctrl`. On that layer's
`done`, advances to `next_instr_addr` and repeats; on `opcode = HALT`,
reports `seq_done`; on a decode/AXI error, reports `seq_error`. Does not
itself touch weights, ifmap or ofmap data — purely control-plane.

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_axi_addr_width` | positive | address width for `program_base_addr`/`next_instr_addr` |
| `g_instr_word_bytes` | positive | must equal `cnn_accel_pkg.c_instr_word_bytes` |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `program_base_addr` | in | `std_ulogic_vector` | from `cnn_accel_csr` |
| `start` | in | `std_ulogic` | pulse from `cnn_accel_csr` |
| `seq_done` | out | `std_ulogic` | pulse to `cnn_accel_csr` |
| `seq_error` | out | `std_ulogic` | pulse to `cnn_accel_csr` |
| `instr_dma_req_m2s` | out | `cnn_accel_pkg.dma_req_m2s_t` | to its `cnn_accel_axi_read_dma` instance |
| `instr_dma_req_s2m` | in | `cnn_accel_pkg.dma_req_s2m_t` | |
| `instr_stream_m2s` | in | `axi_stream_pkg.axi_stream_m2s_t` | instruction bytes from the DMA instance |
| `instr_stream_s2m` | out | `axi_stream_pkg.axi_stream_s2m_t` | |
| `layer_desc_m2s` | out | `cnn_accel_pkg.layer_desc_m2s_t` | to `cnn_accel_layer_ctrl` |
| `layer_desc_s2m` | in | `cnn_accel_pkg.layer_desc_s2m_t` | |
| `layer_done` | in | `std_ulogic` | pulse from `cnn_accel_layer_ctrl` |
| `layer_error` | in | `std_ulogic` | pulse from `cnn_accel_layer_ctrl` (e.g. AXI error during layer execution) |

## Protocols

Internal `dma_req`/AXI4-Stream to its DMA instance; internal
`layer_desc`/done handshake to `cnn_accel_layer_ctrl`. No AXI4/AXI4-Lite
of its own (delegated to the DMA instance).

## Clock/reset

Synchronous active-high `reset_internal`: the program counter is exactly
the kind of state that must be forced back to idle on host abort (per
`doc/cnn_accel_arch.md` "Reset policy"), so this module has no resetless
state.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

FSM: `IDLE -> FETCH -> DECODE -> DISPATCH -> WAIT_LAYER -> (FETCH | DONE |
ERROR)`.

- `IDLE`: on `start`, latch `pc <= program_base_addr`, go to `FETCH`.
- `FETCH`: issue `instr_dma_req` (`addr=pc`, `length=g_instr_word_bytes`);
  on `instr_dma_req_s2m.ready` accepted, collect `g_instr_word_bytes`
  worth of stream beats into a shift register; on the DMA's `dma_done`,
  go to `DECODE`. An AXI error response surfaced by the DMA instance (see
  `cnn_accel_axi_read_dma`'s `resp_error` output) goes straight to
  `ERROR`.
- `DECODE`: unpack the 16-word descriptor into `layer_desc_t` per
  `cnn_accel_pkg`'s byte offsets; an unrecognized `opcode` (not one of the
  six defined values) goes to `ERROR`; `opcode = HALT` goes to `DONE`;
  otherwise go to `DISPATCH`.
- `DISPATCH`: present `layer_desc_m2s` (`valid=1`) until
  `layer_desc_s2m.ready`, then go to `WAIT_LAYER`.
- `WAIT_LAYER`: on `layer_done`, `pc <= next_instr_addr`, go back to
  `FETCH`; on `layer_error`, go to `ERROR`.
- `DONE`/`ERROR`: pulse `seq_done`/`seq_error` for one cycle, return to
  `IDLE`.
