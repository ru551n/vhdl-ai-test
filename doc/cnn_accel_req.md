# CNN Accelerator — Top-Level Requirement (`cnn_accel`)

## Intent

A programmable, memory-mapped FPGA accelerator that executes a **compiled
CNN program**: a sequential list of fixed-width layer instructions produced
by an external Python compiler and stored in DDR alongside weights,
biases, input activations and output activations. The accelerator is not a
fixed network — it is a small in-order engine that fetches one instruction
per layer, configures its datapath accordingly, executes the layer, and
moves on, until a `HALT` instruction is reached.

This mirrors the repo's existing streaming-IP workflow (see
`doc/canny_arch.md`) but targets memory-mapped AXI4 + DDR-resident
tensors/program instead of a live AXI4-Stream video feed, since feature
maps and weights are too large to assume live streaming end-to-end.

## Functional scope (v1)

- Numeric representation: **int8** activations and weights, **int32**
  accumulation.
- Supported layer opcodes: `CONV2D` (arbitrary kernel/stride/padding),
  `DWCONV2D` (depthwise), `POOL_MAX`, `POOL_AVG`, `FC` (fully-connected,
  executed as a degenerate 1x1-spatial `CONV2D`).
- Fused per-layer output stage: bias add (int32) + requantize (multiply +
  arithmetic shift + saturate to int8) + optional ReLU clamp, all driven by
  per-instruction fields — no separate bias/requant/activation opcodes.
- Instructions, weights, biases, input/output activations all live in
  external DDR; the accelerator is an AXI4 master toward DDR (or an
  interconnect in front of it) and an AXI4-Lite slave for host
  control/status.
- On-chip storage is limited to line buffers (sliding-window generation)
  and a double-buffered weight/bias cache per layer tile — never a whole
  feature map.
- Single, in-order instruction stream per invocation (no branches in v1;
  the instruction format reserves an explicit next-instruction-address
  field so branching/looping can be added later without an ISA break).

## Non-goals (v1)

- No on-the-fly quantization/training; the compiler is assumed to have
  already produced int8 weights, biases and requantization
  scale/shift per layer.
- No multi-core / multi-accelerator scaling, no virtual memory (physical
  DDR byte addresses only).
- No dynamic control flow (loops/branches) in the instruction stream.

## External interfaces

- `s_axi_lite` — AXI4-Lite slave, host control/status (start, program base
  address, done/error/IRQ, abort/soft-reset).
- `m_axi` — one AXI4 master port to DDR/interconnect: instruction fetch,
  weight/bias fetch, input-activation fetch (read side, arbitrated 3:1),
  output-activation write-back (write side, single writer).
- `irq` — level or pulse interrupt to host on program done / error.

## Target platform

- FPGA family: Xilinx 7-series / UltraScale (exact device TBD at
  synthesis time). RTL stays `PORTABLE_VHDL` by default; DSP48/BRAM
  inference intent is documented in the architecture doc, no hard IP
  instantiation in v1.
- VHDL-2008, `ieee.std_logic_1164`/`ieee.numeric_std`, unresolved types
  (`std_ulogic`/`std_ulogic_vector`) for all new entities in this IP —
  consistent with `hdl-modules`, which already uses unresolved types
  throughout.

## Verification approach (for later `vhtestgen`/`vhtestrun` phases)

- Python golden model (analogous to `canny_model.py`): a per-opcode
  reference implementation (int8 conv/dwconv/pool/fc + bias/requant/ReLU)
  plus an instruction-stream interpreter, used both for isolated
  per-module VUnit tests and a full-program integration test.
- Reuse `vunit`, `tsfpga`, `hdl-modules` exactly as already wired in
  `run.py`.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

See `doc/cnn_accel_arch.md` for the full architectural decomposition,
instruction-set encoding, submodule responsibilities, and interface/clock/
reset details derived from this requirement.
