# cnn_accel_pkg — requirement

## Responsibility

Package (not an entity): shared record types and constants used across
`modules/cnn_accel/`, so no `HALT`/`CONV2D`/... opcode value or field
layout is duplicated per module.

## Contents

- Opcode constants matching `doc/cnn_accel_arch.md` "Instruction Set
  (v1)": `OPCODE_HALT`, `OPCODE_CONV2D`, `OPCODE_DWCONV2D`,
  `OPCODE_POOL_MAX`, `OPCODE_POOL_AVG`, `OPCODE_FC` (`std_ulogic_vector(7
  downto 0)`).
- Flag bit constants: `FLAG_RELU_EN`, `FLAG_BIAS_EN`, `FLAG_REQUANT_EN`,
  `FLAG_PAD_EN` (bit indices into the W0 flags byte).
- `layer_desc_t`: decoded-instruction record (one field per ISA table
  column: opcode, flags, in_addr, out_addr, weight_addr, bias_addr,
  in_width, in_height, in_channels, out_channels, kernel_h/w, stride_h/w,
  pad_top/bottom/left/right, requant_scale, requant_shift, pool_kernel_h/w,
  pool_stride_h/w, next_instr_addr) plus `layer_desc_m2s_t` (`valid` +
  `layer_desc_t`) / `layer_desc_s2m_t` (`ready`) handshake wrapper records,
  per `shared/InterfaceRecords.md`.
- `dma_req_t` / `dma_req_m2s_t` (`valid`,`addr`,`length`) /
  `dma_req_s2m_t` (`ready`) and a `dma_done : std_ulogic` convention used
  by every `cnn_accel_axi_read_dma`/`cnn_accel_ofmap_dma` request port.
- `c_instr_word_bytes : positive := 64` and per-word byte-offset
  constants (`c_off_opcode`, `c_off_in_addr`, ...) matching the ISA table,
  so `cnn_accel_sequencer`'s decode logic and any golden-model/testbench
  encoder agree on layout by construction.

## Clock/reset

N/A (package, no clock/reset).

<!-- functional-spec: hand-owned below this line -->

## Functional Description

Pure constant/type declarations; no behavior. Authoritative source for
the ISA field layout — `doc/cnn_accel_arch.md`'s table must stay in sync
with this package's byte offsets on any future revision.
