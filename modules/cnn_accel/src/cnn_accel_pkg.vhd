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

  ------------------------------------------------------------------------
  -- Window link: cnn_accel_window_gen -> pe_array. One K_h x K_w x
  -- tile_channels window of int8 activations per beat, for a runtime
  -- generic-sized group ("tile") of input channels. 'data' is
  -- unconstrained here and must be constrained at the declaration site
  -- (signal, port, etc.), sized via window_data_length() below, per
  -- shared/InterfaceRecords.md's unconstrained-element pattern.
  --
  -- Element layout of 'data': tap 'i' (row-major, i = row * kernel_w +
  -- col), channel 'c' within the tile, is element 'i*tile_channels + c'.
  --
  -- 'data' is an *array of int8 elements*, not one wide packed vector
  -- (D15). The consumer indexes it with a runtime tap/group counter, and
  -- a runtime-indexed slice of a packed vector is a dynamic slice, which
  -- GHDL's synthesis backend rejects outright ("cannot extract same
  -- variable part for dynamic slice") while simulating perfectly happily
  -- -- the same trap already worked around by hand in
  -- cnn_accel_window_gen.vhd and cnn_accel_weight_buffer.vhd. An array
  -- index is a first-class index: the tool infers the multiplexer and
  -- nothing has to know the element width. See shared/ModernVHDL.md,
  -- "Aggregate representation". Use to_slv/to_tap_array below only at
  -- boundaries that genuinely need flat bits (AXI payloads, testbench
  -- queues).
  --
  -- first_tile/last_tile: for an output pixel needing
  -- T = ceil(in_channels / tile_channels) tiles, the producer emits T
  -- consecutive beats, with first_tile = '1' on beat 0 and last_tile = '1'
  -- on beat T-1 (both '1' when T = 1). The consumer clears its accumulator
  -- on first_tile and emits its result on last_tile.
  ------------------------------------------------------------------------

  -- One int8 activation, and a run of them. The element width is fixed at
  -- 8 (the accelerator's activation type, D-arithmetic contract), so this
  -- array needs only its index range constrained at the declaration site.
  subtype tap_t is std_ulogic_vector(7 downto 0);
  type tap_array_t is array (natural range <>) of tap_t;

  type window_m2s_t is record
    valid      : std_ulogic;
    last       : std_ulogic;  -- last output pixel of the feature map
    first_tile : std_ulogic;  -- first input-channel tile of this output pixel
    last_tile  : std_ulogic;  -- last input-channel tile of this output pixel
    data       : tap_array_t;
  end record;

  type window_s2m_t is record
    ready : std_ulogic;
  end record;

  ------------------------------------------------------------------------
  -- Accumulator link: pe_array -> bias_requant. One int32-ish partial sum
  -- per PE row (output channel). Like the window link (D15) this is an
  -- array of lanes rather than a packed vector, so a lane can be selected
  -- by a runtime index without a dynamic slice.
  --
  -- Unlike 'tap_array_t' the element width is *not* fixed -- it follows
  -- 'g_accum_width' -- so both the index range and the element range must
  -- be constrained at the declaration site (VHDL-2008 array-of-
  -- unconstrained-element):
  --
  --   signal m_accum_m2s : accum_m2s_t(data(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0));
  ------------------------------------------------------------------------

  type accum_array_t is array (natural range <>) of signed;

  type accum_m2s_t is record
    valid : std_ulogic;
    last  : std_ulogic;
    data  : accum_array_t;  -- one partial sum per PE row
  end record;

  type accum_s2m_t is record
    ready : std_ulogic;
  end record;

  -- Number of int8 *elements* in window_m2s_t.data for the given kernel
  -- size (K_h = K_w = kernel_size) and tile_channels, per the element
  -- layout documented above. Constrain the record as
  -- 'window_m2s_t(data(0 to window_data_length(...) - 1))'.
  function window_data_length(kernel_size : positive; tile_channels : positive) return positive;

  -- Flat-bit views of a tap array, for the boundaries that still need
  -- packed bits (AXI payloads, VUnit queue push/pop). Element 'i' of the
  -- array occupies bits '8*i + 7 downto 8*i' of the vector, so this is
  -- exactly the pre-D15 packing. Do not use these to work around indexing
  -- inside RTL -- that is what the array is for.
  function to_slv(data : tap_array_t) return std_ulogic_vector;
  function to_tap_array(data : std_ulogic_vector) return tap_array_t;

end package cnn_accel_pkg;

package body cnn_accel_pkg is

  function window_data_length(kernel_size : positive; tile_channels : positive) return positive is
  begin
    return kernel_size * kernel_size * tile_channels;
  end function;

  function to_slv(data : tap_array_t) return std_ulogic_vector is
    variable result : std_ulogic_vector(8 * data'length - 1 downto 0);
    variable idx : natural := 0;
  begin
    for i in data'range loop
      result(8 * (idx + 1) - 1 downto 8 * idx) := data(i);
      idx := idx + 1;
    end loop;
    return result;
  end function;

  function to_tap_array(data : std_ulogic_vector) return tap_array_t is
    variable normalized : std_ulogic_vector(data'length - 1 downto 0) := data;
    variable result : tap_array_t(0 to data'length / 8 - 1);
  begin
    for i in result'range loop
      result(i) := normalized(8 * (i + 1) - 1 downto 8 * i);
    end loop;
    return result;
  end function;

end package body cnn_accel_pkg;
