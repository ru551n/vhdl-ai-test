library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- ISA v2.0 / storage-model constants and the v2 decoded-descriptor record,
-- per doc/cnn_accel_top_v2_arch.md sections 3, 5 and 8. v1.2 opcodes are
-- re-exported (aliased, never redefined with a new literal) from the
-- generated 'cnn_accel_isa_pkg' so a consumer of only this package still
-- sees every opcode under the 'c_opcode_*' naming convention used here.
-- Every byte offset, opcode, flag bit and space-tag value comes from the
-- generated 'cnn_accel_isa_pkg' ('c_off_*', 'OPCODE_*', 'FLAG_*',
-- 'SPACE_*', 'SPACE_SHIFT_*'), whose single source of truth is
-- 'cnn_accel_constants.py' -- nothing about the instruction word is
-- restated with a literal here.
library cnn_accel;
use cnn_accel.cnn_accel_isa_pkg.all;
use cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_isa_version;

package cnn_accel_v2_pkg is

  ------------------------------------------------------------------------
  -- Storage model (spec section 3).
  ------------------------------------------------------------------------

  subtype space_t is std_ulogic_vector(1 downto 0);

  constant c_space_ddr          : space_t := SPACE_DDR;
  constant c_space_local_tensor : space_t := SPACE_LOCAL_TENSOR;
  constant c_space_local_weight : space_t := SPACE_LOCAL_WEIGHT;
  constant c_space_reserved     : space_t := SPACE_RESERVED;

  ------------------------------------------------------------------------
  -- ISA v2.0 opcodes (spec section 5.2). 'c_opcode_halt' .. 'c_opcode_fc'
  -- are aliases of the generated v1.2 'OPCODE_*' constants (same value,
  -- single source of truth in cnn_accel_constants.py); 'c_opcode_load' ..
  -- 'c_opcode_act' are new in v2.0 and have no generated counterpart yet.
  ------------------------------------------------------------------------

  constant c_opcode_halt     : std_ulogic_vector(7 downto 0) := OPCODE_HALT;
  constant c_opcode_conv2d   : std_ulogic_vector(7 downto 0) := OPCODE_CONV2D;
  constant c_opcode_dwconv2d : std_ulogic_vector(7 downto 0) := OPCODE_DWCONV2D;
  constant c_opcode_pool_max : std_ulogic_vector(7 downto 0) := OPCODE_POOL_MAX;
  constant c_opcode_pool_avg : std_ulogic_vector(7 downto 0) := OPCODE_POOL_AVG;
  constant c_opcode_fc       : std_ulogic_vector(7 downto 0) := OPCODE_FC;

  constant c_opcode_load     : std_ulogic_vector(7 downto 0) := OPCODE_LOAD;
  constant c_opcode_store    : std_ulogic_vector(7 downto 0) := OPCODE_STORE;
  constant c_opcode_loadw    : std_ulogic_vector(7 downto 0) := OPCODE_LOADW;
  constant c_opcode_add      : std_ulogic_vector(7 downto 0) := OPCODE_ADD;
  constant c_opcode_upsample : std_ulogic_vector(7 downto 0) := OPCODE_UPSAMPLE;
  constant c_opcode_copy     : std_ulogic_vector(7 downto 0) := OPCODE_COPY;
  constant c_opcode_act      : std_ulogic_vector(7 downto 0) := OPCODE_ACT;

  ------------------------------------------------------------------------
  -- Flag bit indices (W0 bits [15:8], spec section 5.1). 'c_flag_relu_en'
  -- .. 'c_flag_per_channel_en' alias the generated v1.2 'FLAG_*' constants;
  -- 'c_flag_act_lut_en'/'c_flag_weight_reuse' are new in v2.0.
  ------------------------------------------------------------------------

  constant c_flag_relu_en        : natural := FLAG_RELU_EN;
  constant c_flag_bias_en        : natural := FLAG_BIAS_EN;
  constant c_flag_requant_en     : natural := FLAG_REQUANT_EN;
  constant c_flag_pad_en         : natural := FLAG_PAD_EN;
  constant c_flag_clamp_en       : natural := FLAG_CLAMP_EN;
  constant c_flag_per_channel_en : natural := FLAG_PER_CHANNEL_EN;
  constant c_flag_act_lut_en     : natural := FLAG_ACT_LUT_EN;
  constant c_flag_weight_reuse   : natural := FLAG_WEIGHT_REUSE;

  ------------------------------------------------------------------------
  -- Absolute bit position of each operand's 2-bit space tag inside the
  -- 512-bit (little-endian) instruction word: the generated 'c_off_spaces'
  -- gives the byte the tags are packed into (W0 byte 2), and the generated
  -- 'SPACE_SHIFT_*' gives each tag's bit index within that byte. Composing
  -- the two here keeps the slicing in 'decode_desc_v2' readable without
  -- restating either number.
  ------------------------------------------------------------------------

  constant c_bit_spaces : natural := 8 * c_off_spaces;

  constant c_bit_space_src0 : natural := c_bit_spaces + SPACE_SHIFT_SRC0;
  constant c_bit_space_src1 : natural := c_bit_spaces + SPACE_SHIFT_SRC1;
  constant c_bit_space_dst  : natural := c_bit_spaces + SPACE_SHIFT_DST;
  constant c_bit_space_wgt  : natural := c_bit_spaces + SPACE_SHIFT_WGT;

  ------------------------------------------------------------------------
  -- Error codes (spec section 9), 4 bits, as they appear in
  -- CSR.STATUS[7:4].
  ------------------------------------------------------------------------

  subtype err_code_t is std_ulogic_vector(3 downto 0);

  constant c_err_none            : err_code_t := x"0";
  constant c_err_unsupported_op  : err_code_t := x"1";
  constant c_err_bad_space       : err_code_t := x"2";
  constant c_err_misaligned      : err_code_t := x"3";
  constant c_err_local_range     : err_code_t := x"4";
  constant c_err_ddr_range       : err_code_t := x"5";
  constant c_err_bad_reserved    : err_code_t := x"6";
  constant c_err_bad_geometry    : err_code_t := x"7";
  constant c_err_axi             : err_code_t := x"8";
  constant c_err_timeout         : err_code_t := x"9";

  -- Reported to the host in HW_INFO2.ISA_VERSION. Derived from the same
  -- generated constant 'cnn_accel_csr' drives that register from, so the
  -- version this package's decoder implements and the version the host
  -- reads out cannot disagree.
  constant c_isa_version : std_ulogic_vector(15 downto 0) :=
    std_ulogic_vector(to_unsigned(cnn_accel_constant_isa_version, 16));

  ------------------------------------------------------------------------
  -- Decoded ISA v2.0 descriptor. Every field of 'cnn_accel_pkg.layer_desc_t'
  -- is reproduced verbatim (same name, same type) plus the v2.0 additions:
  -- the four operand space tags and 'xfer_bytes' (which doubles as
  -- 'src1_addr' for ADD, same bits -- spec section 5.1, W15), plus the two
  -- 'reserved, must be 0' gaps. The reserved gaps are carried as real
  -- record fields for one reason: section 9 requires 'ERR_BAD_RESERVED',
  -- and a validator that cannot see the bits it is supposed to police
  -- would silently pass every malformed program. Keeping them here rather
  -- than exposing a raw instruction word from 'cnn_accel_cmd_fetch' keeps
  -- 'decode_desc_v2' the single place that knows the word layout.
  ------------------------------------------------------------------------

  type desc_v2_t is record
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
    output_offset   : signed(15 downto 0);
    clamp_min       : signed(7 downto 0);
    clamp_max       : signed(7 downto 0);
    scale_addr      : unsigned(31 downto 0);
    -- ISA v2.0 additions (spec section 5.1).
    space_src0      : space_t;
    space_src1      : space_t;
    space_dst       : space_t;
    space_wgt       : space_t;
    xfer_bytes      : unsigned(31 downto 0);
    -- W0 byte 3 and W10 bytes 41-43: must be zero (section 5.1).
    reserved_w0     : std_ulogic_vector(7 downto 0);
    reserved_w10    : std_ulogic_vector(23 downto 0);
  end record;

  -- All-zero descriptor (opcode HALT, space DDR everywhere), useful as a
  -- reset/idle value and as a base for testbench-style field overrides.
  function desc_v2_init return desc_v2_t;

  -- Decode one 64-byte (512-bit) descriptor, byte 0 in bits [7:0], i.e.
  -- 'data(8*n+7 downto 8*n)' is descriptor byte 'n' (spec section 5.1).
  function decode_desc_v2(data : std_ulogic_vector(511 downto 0)) return desc_v2_t;

  ------------------------------------------------------------------------
  -- Performance counters (spec section 8, CSR offsets 0x18-0x3C), grouped
  -- into a single record so 'cnn_accel_csr' takes one port instead of
  -- eleven. Each field is the free-running (or accumulating) counter
  -- value as maintained by whichever module owns it; 'cnn_accel_csr'
  -- itself only samples these into the read-only registers -- it is not
  -- the counters' owner. 'local_rd_kib'/'local_wr_kib' are full 32-bit
  -- counts; 'cnn_accel_csr' packs their low 16 bits into the combined
  -- 'LOCAL_RD_BYTES'/'LOCAL_WR_BYTES' register ([15:0]/[31:16]) per the
  -- spec's KiB-granularity field.
  ------------------------------------------------------------------------

  type csr_counters_t is record
    cmd_count         : std_ulogic_vector(31 downto 0);
    cycle_count       : std_ulogic_vector(31 downto 0);
    compute_cycles    : std_ulogic_vector(31 downto 0);
    stall_cycles      : std_ulogic_vector(31 downto 0);
    ddr_rd_bytes      : std_ulogic_vector(31 downto 0);
    ddr_wr_bytes      : std_ulogic_vector(31 downto 0);
    tensor_load_count : std_ulogic_vector(31 downto 0);
    tensor_store_count : std_ulogic_vector(31 downto 0);
    weight_load_bytes : std_ulogic_vector(31 downto 0);
    local_rd_kib      : std_ulogic_vector(31 downto 0);
    local_wr_kib      : std_ulogic_vector(31 downto 0);
  end record;

  constant csr_counters_init : csr_counters_t := (
    cmd_count           => (others => '0'),
    cycle_count         => (others => '0'),
    compute_cycles      => (others => '0'),
    stall_cycles        => (others => '0'),
    ddr_rd_bytes        => (others => '0'),
    ddr_wr_bytes        => (others => '0'),
    tensor_load_count   => (others => '0'),
    tensor_store_count  => (others => '0'),
    weight_load_bytes   => (others => '0'),
    local_rd_kib        => (others => '0'),
    local_wr_kib        => (others => '0')
  );

end package cnn_accel_v2_pkg;

package body cnn_accel_v2_pkg is

  function desc_v2_init return desc_v2_t is
    constant result : desc_v2_t := (
      opcode          => (others => '0'),
      flags           => (others => '0'),
      in_addr         => (others => '0'),
      out_addr        => (others => '0'),
      weight_addr     => (others => '0'),
      bias_addr       => (others => '0'),
      in_width        => (others => '0'),
      in_height       => (others => '0'),
      in_channels     => (others => '0'),
      out_channels    => (others => '0'),
      kernel_h        => (others => '0'),
      kernel_w        => (others => '0'),
      stride_h        => (others => '0'),
      stride_w        => (others => '0'),
      pad_top         => (others => '0'),
      pad_bottom      => (others => '0'),
      pad_left        => (others => '0'),
      pad_right       => (others => '0'),
      requant_scale   => (others => '0'),
      requant_shift   => (others => '0'),
      pool_kernel_h   => (others => '0'),
      pool_kernel_w   => (others => '0'),
      pool_stride_h   => (others => '0'),
      pool_stride_w   => (others => '0'),
      next_instr_addr => (others => '0'),
      output_offset   => (others => '0'),
      clamp_min       => (others => '0'),
      clamp_max       => (others => '0'),
      scale_addr      => (others => '0'),
      space_src0      => (others => '0'),
      space_src1      => (others => '0'),
      space_dst       => (others => '0'),
      space_wgt       => (others => '0'),
      xfer_bytes      => (others => '0'),
      reserved_w0     => (others => '0'),
      reserved_w10    => (others => '0')
    );
  begin
    return result;
  end function;

  function decode_desc_v2(data : std_ulogic_vector(511 downto 0)) return desc_v2_t is
    variable result : desc_v2_t := desc_v2_init;

    -- Byte 'n' (0 to 63) of the descriptor, per the little-endian mapping
    -- documented on the 'data' parameter above.
    function byte(n : natural) return std_ulogic_vector is
    begin
      return data(8 * n + 7 downto 8 * n);
    end function;

    -- Bits [hi:lo] of the descriptor's flat bit view; 'off_bytes' is the
    -- generated/local byte offset ('c_off_*') of the field's first byte.
    function field(off_bytes : natural; width_bits : positive) return std_ulogic_vector is
    begin
      return data(8 * off_bytes + width_bits - 1 downto 8 * off_bytes);
    end function;
  begin
    result.opcode          := byte(c_off_opcode);
    result.flags           := byte(c_off_flags);

    result.space_src0      := data(c_bit_space_src0 + 1 downto c_bit_space_src0);
    result.space_src1      := data(c_bit_space_src1 + 1 downto c_bit_space_src1);
    result.space_dst       := data(c_bit_space_dst + 1 downto c_bit_space_dst);
    result.space_wgt       := data(c_bit_space_wgt + 1 downto c_bit_space_wgt);

    result.in_addr         := unsigned(field(c_off_in_addr, 32));
    result.out_addr        := unsigned(field(c_off_out_addr, 32));
    result.weight_addr     := unsigned(field(c_off_weight_addr, 32));
    result.bias_addr       := unsigned(field(c_off_bias_addr, 32));

    result.in_width        := unsigned(field(c_off_in_width, 16));
    result.in_height       := unsigned(field(c_off_in_height, 16));
    result.in_channels     := unsigned(field(c_off_in_channels, 16));
    result.out_channels    := unsigned(field(c_off_out_channels, 16));

    result.kernel_h        := unsigned(field(c_off_kernel_h, 8));
    result.kernel_w        := unsigned(field(c_off_kernel_w, 8));
    result.stride_h        := unsigned(field(c_off_stride_h, 8));
    result.stride_w        := unsigned(field(c_off_stride_w, 8));

    result.pad_top         := unsigned(field(c_off_pad_top, 8));
    result.pad_bottom      := unsigned(field(c_off_pad_bottom, 8));
    result.pad_left        := unsigned(field(c_off_pad_left, 8));
    result.pad_right       := unsigned(field(c_off_pad_right, 8));

    result.requant_scale   := signed(field(c_off_requant_scale, 32));
    result.requant_shift   := unsigned(field(c_off_requant_shift, 8));

    result.pool_kernel_h   := unsigned(field(c_off_pool_kernel_h, 8));
    result.pool_kernel_w   := unsigned(field(c_off_pool_kernel_w, 8));
    result.pool_stride_h   := unsigned(field(c_off_pool_stride_h, 8));
    result.pool_stride_w   := unsigned(field(c_off_pool_stride_w, 8));

    result.next_instr_addr := unsigned(field(c_off_next_instr_addr, 32));

    result.output_offset   := signed(field(c_off_output_offset, 16));
    result.clamp_min       := signed(field(c_off_clamp_min, 8));
    result.clamp_max       := signed(field(c_off_clamp_max, 8));

    result.scale_addr      := unsigned(field(c_off_scale_addr, 32));

    result.xfer_bytes      := unsigned(field(c_off_xfer_bytes, 32));

    -- The two reserved gaps. Their positions are derived, not stated: W0
    -- byte 3 is the byte after the 'spaces' tag byte, and the W10 gap is
    -- the three bytes after 'requant_shift'. If a future ISA revision
    -- turns either gap into a real field, the generator stops emitting it
    -- as a gap and these two lines are what must be revisited.
    result.reserved_w0     := field(c_off_spaces + 1, 8);
    result.reserved_w10    := field(c_off_requant_shift + 1, 24);

    return result;
  end function;

end package body cnn_accel_v2_pkg;
