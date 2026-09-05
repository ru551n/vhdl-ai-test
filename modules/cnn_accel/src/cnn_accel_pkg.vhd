library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Shared record types and constants for modules/cnn_accel/. See
-- modules/cnn_accel/doc/cnn_accel_pkg_req.md and doc/cnn_accel_arch.md
-- ("Instruction Set (v1)") for the authoritative field layout. Pure
-- constant/type declarations; no behavior.
package cnn_accel_pkg is

  ------------------------------------------------------------------------
  -- Opcodes (W0 bits [7:0])
  ------------------------------------------------------------------------

  constant OPCODE_HALT     : std_ulogic_vector(7 downto 0) := x"00";
  constant OPCODE_CONV2D   : std_ulogic_vector(7 downto 0) := x"01";
  constant OPCODE_DWCONV2D : std_ulogic_vector(7 downto 0) := x"02";
  constant OPCODE_POOL_MAX : std_ulogic_vector(7 downto 0) := x"03";
  constant OPCODE_POOL_AVG : std_ulogic_vector(7 downto 0) := x"04";
  constant OPCODE_FC       : std_ulogic_vector(7 downto 0) := x"05";

  ------------------------------------------------------------------------
  -- Flag bits (W0 bits [15:8]), indices into the flags byte
  ------------------------------------------------------------------------

  constant FLAG_RELU_EN    : natural := 0;
  constant FLAG_BIAS_EN    : natural := 1;
  constant FLAG_REQUANT_EN : natural := 2;
  constant FLAG_PAD_EN     : natural := 3;

  ------------------------------------------------------------------------
  -- Instruction word layout: size and per-field byte offsets, matching
  -- doc/cnn_accel_arch.md's ISA table. cnn_accel_model.py's encoder must
  -- agree with these byte-for-byte.
  ------------------------------------------------------------------------

  constant c_instr_word_bytes : positive := 64;

  constant c_off_opcode         : natural := 0;
  constant c_off_flags          : natural := 1;
  -- W0 bytes 2-3: reserved, must be 0.
  constant c_off_in_addr         : natural := 4;
  constant c_off_out_addr        : natural := 8;
  constant c_off_weight_addr     : natural := 12;
  constant c_off_bias_addr       : natural := 16;
  constant c_off_in_width        : natural := 20;
  constant c_off_in_height       : natural := 22;
  constant c_off_in_channels     : natural := 24;
  constant c_off_out_channels    : natural := 26;
  constant c_off_kernel_h        : natural := 28;
  constant c_off_kernel_w        : natural := 29;
  constant c_off_stride_h        : natural := 30;
  constant c_off_stride_w        : natural := 31;
  constant c_off_pad_top         : natural := 32;
  constant c_off_pad_bottom      : natural := 33;
  constant c_off_pad_left        : natural := 34;
  constant c_off_pad_right       : natural := 35;
  constant c_off_requant_scale   : natural := 36;
  constant c_off_requant_shift   : natural := 40;
  -- W10 bytes 41-43: reserved.
  constant c_off_pool_kernel_h   : natural := 44;
  constant c_off_pool_kernel_w   : natural := 45;
  constant c_off_pool_stride_h   : natural := 46;
  constant c_off_pool_stride_w   : natural := 47;
  constant c_off_next_instr_addr : natural := 48;
  -- W13-W15 bytes 52-63: reserved, must be 0.

  ------------------------------------------------------------------------
  -- Decoded-instruction record, one field per ISA table column.
  ------------------------------------------------------------------------

  type layer_desc_t is record
    opcode          : std_ulogic_vector(7 downto 0);
    flags           : std_ulogic_vector(7 downto 0);
    in_addr         : unsigned(31 downto 0);
    out_addr        : unsigned(31 downto 0);
    weight_addr     : unsigned(31 downto 0);
    bias_addr       : unsigned(31 downto 0);
    in_width        : unsigned(15 downto 0);
    in_height       : unsigned(15 downto 0);
    in_channels     : unsigned(15 downto 0);
    out_channels    : unsigned(15 downto 0);
    kernel_h        : unsigned(7 downto 0);
    kernel_w        : unsigned(7 downto 0);
    stride_h        : unsigned(7 downto 0);
    stride_w        : unsigned(7 downto 0);
    pad_top         : unsigned(7 downto 0);
    pad_bottom      : unsigned(7 downto 0);
    pad_left        : unsigned(7 downto 0);
    pad_right       : unsigned(7 downto 0);
    requant_scale   : signed(31 downto 0);
    requant_shift   : unsigned(7 downto 0);
    pool_kernel_h   : unsigned(7 downto 0);
    pool_kernel_w   : unsigned(7 downto 0);
    pool_stride_h   : unsigned(7 downto 0);
    pool_stride_w   : unsigned(7 downto 0);
    next_instr_addr : unsigned(31 downto 0);
  end record;

  -- Handshake wrapper records, per shared/InterfaceRecords.md.
  type layer_desc_m2s_t is record
    valid : std_ulogic;
    desc  : layer_desc_t;
  end record;

  type layer_desc_s2m_t is record
    ready : std_ulogic;
  end record;

  ------------------------------------------------------------------------
  -- Generic DMA request handshake, shared by cnn_accel_axi_read_dma and
  -- cnn_accel_ofmap_dma request ports. Every such request port also
  -- carries a separate "dma_done : std_ulogic" pulse output, by
  -- convention, once the requested transfer has completed on the AXI
  -- side; that pulse is not part of the request/response handshake
  -- itself and so is not modeled as a record field here.
  ------------------------------------------------------------------------

  type dma_req_t is record
    addr   : unsigned(31 downto 0);
    length : unsigned(31 downto 0);
  end record;

  type dma_req_m2s_t is record
    valid : std_ulogic;
    req   : dma_req_t;
  end record;

  type dma_req_s2m_t is record
    ready : std_ulogic;
  end record;

end package cnn_accel_pkg;
