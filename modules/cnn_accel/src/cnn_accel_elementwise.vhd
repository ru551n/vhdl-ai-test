library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;
use cnn_accel.cnn_accel_v2_pkg.all;
use cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_activation_plane_channels;

-- ISA v2.0 "local-to-local" compute engine (doc/cnn_accel_top_v2_arch.md
-- section 5.2, "cnn_accel_elementwise"): 'COPY', 'ACT', 'ADD' and
-- 'UPSAMPLE'. Every operand this entity touches lives in
-- 'cnn_accel_tensor_mem' (LOCAL_TENSOR) or the LOCAL_WEIGHT-backed LUT
-- store -- unlike 'cnn_accel_conv_core', this entity owns no DDR-facing
-- AXI4 port at all, and unlike 'cnn_accel_cmd_fetch' it is driven by
-- already-decoded scalar fields (mirroring 'cnn_accel_conv_core's
-- 'cfg_*'-port convention), not a whole 'desc_v2_t' record.
--
-- Port-to-channel contract (section 4's table), for 'cmd_proc' to wire up
-- directly, one dma_req_m2s_t/dma_req_s2m_t + one AXI4-Stream pair per
-- channel, exactly as 'cnn_accel_tensor_mem' exposes them:
--   'src0_*' -> tensor_mem 'r0', 'src1_*' -> tensor_mem 'r1' (ADD only),
--   'dst_*' -> tensor_mem 'w1'.
-- 'lut_*' is this entity's own fourth channel, added by vhdesign: the
-- ISA (section 5.1/5.2, 'cnn_accel_constants.py's 'ACT_LUT_EN' comment)
-- places ACT's 256-entry table "in LOCAL_WEIGHT" but defines no dedicated
-- LUT-address field and no byte-addressable read port for it --
-- 'cnn_accel_weight_buffer's existing regions (weight/bias/scale) are
-- tile-row oriented (one 'g_pe_rows*g_pe_cols'-lane row per read address),
-- not indexable by a single int8 value 0..255, and extending it is out of
-- this entity's scope. 'lut_req_*'/'s_lut_stream_*' therefore requests a
-- plain linear read burst (same dma_req_m2s_t/axi_stream_m2s_t idiom as
-- every other channel here) of whatever LOCAL_WEIGHT-space memory
-- 'cmd_proc'/'cnn_accel_top' ultimately backs it with; wiring that memory
-- is integration, not this entity's concern.
--
-- Resolved ambiguity (ACT's LUT base address): the ISA v2.0 field table
-- has no field named for it. 'weight_addr' (W3, space 'space_wgt') is the
-- only address field tagged with a LOCAL_WEIGHT-capable space that ACT
-- does not otherwise need (ACT has no weights/bias/scale of its own), so
-- this entity reads ACT's LUT base from the descriptor's 'weight_addr' --
-- the same field CONV2D's weight tensor uses, reused for a different
-- opcode exactly like 'xfer_bytes'/'src1_addr' already alias W15 for ADD.
-- 'accel_v2.planner'/'accel_v2.isa' do not yet lower 'ActOp' (still WIP,
-- Python-side), so there is no independent existing convention to check
-- this against; if a future planner disagrees, only this one comment and
-- the 'lut_addr' port binding need to change.
--
-- Resolved ambiguity (ADD's transfer size): section 5.2's opcode table
-- lists ADD as "xfer_bytes long", but 'cnn_accel_constants.py's ISA_LAYOUT
-- comment for W15 is explicit that "for ADD these same bits instead carry
-- the second source operand's address (src1_addr), since ADD's size is
-- implied by its tensor geometry" -- and 'accel_v2.model.AddOp' indeed
-- carries no 'xfer_bytes' field at all, only 'requant_scale'/
-- 'requant_shift', with its transfer size coming from its (shared)
-- input/output 'Tensor' shape like every other op. The opcode table's
-- wording is therefore stale (most likely copy-pasted from COPY's row);
-- this entity follows the authoritative field-table comment and the
-- Python model: ADD's byte count is computed from 'in_width'/'in_height'/
-- 'in_channels' (decision-S6 padded-channel byte count, see below), and
-- 'xfer_bytes' as a port is only consulted for 'COPY'/'ACT'.
--
-- Channel-tiled plane layout (decision S6, doc/cnn_accel_top_v2_arch.md
-- section 3 and 'cnn_accel_model.pack_activation_planes'): an activation
-- is stored as '[C/T][H][W][T]' planes, T =
-- 'cnn_accel_constant_activation_plane_channels' (a generated constant,
-- never hand-typed here), each occupying
-- 'ceil(channels/T) * T * width * height' bytes (the zero-padded last
-- plane still occupies bytes). 'ADD' and 'ACT'/'COPY' never need to
-- decode this layout: an elementwise byte-for-byte operation (or an
-- opaque copy) over two/one tensors that share the exact same physical
-- layout is layout-agnostic by construction -- only the total (padded)
-- byte count matters, which this entity computes with the same formula
-- 'cnn_accel_model.activation_bytes' uses. 'UPSAMPLE' is the one opcode
-- that must actually understand the tiling (see its own comment below).
--
-- Design simplification (asserted, not merely assumed): this entity
-- processes one full AXI4-Stream beat ('g_axi_data_width' bits) per
-- internal step, on all four channels, with no byte-serial
-- (dis)assembly. This only produces the right behaviour if one beat is
-- exactly one T-byte pixel-plane slice, i.e. 'g_axi_data_width = 8*T' --
-- which is already an architectural invariant of this project
-- ('cnn_accel_constants.py': "MAX_AXI_DATA_WIDTH = 8*ACTIVATION_PLANE_
-- CHANNELS", and 'cnn_accel_tensor_mem's 'g_data_width' is "fixed = AXI
-- data width" at that same value), not a new constraint invented here.
-- The entity-level assertion below turns any future violation into an
-- immediate elaboration failure instead of a silent wrong-data bug.
--
-- Per-channel completion: every channel here follows the same linear
-- '(base, length_bytes)' burst contract 'cnn_accel_tensor_mem' documents
-- (section 4): the requested length always agrees with the number of
-- beats this entity actually streams, and the final beat's AXI4-Stream
-- 'last' is therefore the authoritative "this burst is fully exchanged"
-- signal. This entity relies on 'last' alone and does not have (or need)
-- a 'src0_done'/'src1_done'/'dst_done'/'lut_done' input: tensor_mem's own
-- per-channel 'done' pulses exist for its own bank-arbitration bookkeeping
-- (see its own header comment), not because the requesting side needs a
-- second completion signal on top of 'last'. 'cmd_proc' simply leaves
-- tensor_mem's 'r0_done'/'r1_done'/'w1_done' outputs unconnected when
-- wiring this entity, exactly as it would for any unused output port.
--
-- Requests are issued strictly one channel at a time (never two
-- 'dma_req_m2s_t.valid's asserted together), even for 'ADD' (src0, then
-- src1, then dst) -- so a single shared address/length register pair
-- drives whichever '*_req_m2s' is currently asserted, and no per-channel
-- "already accepted" bookkeeping is needed: each request-issuing state is
-- visited exactly once per command and is left the same cycle its
-- request is accepted. The corresponding AXI4-Stream data channels are
-- only ever driven concurrently once every request they belong to has
-- already been posted (e.g. 'ADD' streams src0/src1/dst together only
-- after all three requests were accepted), which every well-behaved
-- AXI4-Stream source (tensor_mem included) must tolerate (holding 'valid'
-- until this entity is ready to consume, per shared/Axi4.md).
--
-- 'done' pulses exactly once per command, on both the success and the
-- error path (resolved ambiguity: 'cmd_proc' needs one uniform
-- "this command is finished, move on" signal regardless of outcome;
-- 'error'/'error_code' are qualifiers valid only on that same cycle, and
-- read '0'/'c_err_none' otherwise). Degenerate inputs ('xfer_bytes = 0',
-- any zero geometry dimension, a non-beat-aligned 'xfer_bytes', or a
-- computed transfer exceeding 'g_max_xfer_bytes') are all reported as
-- 'c_err_bad_geometry' with an immediate 'done' -- no DMA request is ever
-- issued for a command this entity has already determined is malformed.
-- An unrecognized opcode is reported as 'c_err_unsupported_op' the same
-- way. Reset is synchronous and relied on alone to abort an in-flight
-- command (no 'soft_reset' port), the same convention 'cnn_accel_cmd_fetch'
-- and 'cnn_accel_axi_read_dma' already use.
--
-- ADD's shared arithmetic contract with 'cnn_accel_bias_requant': per
-- 'accel_v2.reference._exec_add' (the trusted second implementation) and
-- 'cnn_accel_model.round_shift_right_signed'/'saturate_signed', ADD is
-- 'dst = sat_i8(round_shift_right_signed(src0*scale, 15+shift) +
-- round_shift_right_signed(src1*scale, 15+shift))' -- the *same*
-- round-half-up rescale 'cnn_accel_bias_requant' performs per its own
-- stage 5 ("Verified bit-exact against cnn_accel_model.py's
-- 'round_shift_right_signed'..."), with one shared '(scale, shift)' pair
-- applied identically to both operands (the descriptor carries only one
-- pair for ADD). Reusing 'cnn_accel_bias_requant' itself via instantiation
-- is *not* possible here: that entity's pipeline always ends by
-- saturating its own per-lane result to int8 (its 'sat_result_l', stage
-- 7) before it can be read back out -- there is no tap point for the
-- wide, *pre-saturation* rescaled quotient ADD needs to sum before its
-- own, single, final int8 saturate. Summing two already-saturated int8
-- values would silently diverge from the golden model whenever either
-- operand's rescaled value would have overflowed int8 on its own. This
-- entity therefore reimplements only the round-half-up guard-bit
-- technique itself (function 'round_shift_right_signed' below, the exact
-- same arithmetic as 'cnn_accel_bias_requant.vhd's stage-5 comment,
-- unpipelined and without the bias/offset terms ADD has no use for), and
-- reuses 'math.saturate_signed' -- the same primitive
-- 'cnn_accel_bias_requant' itself instantiates for every saturate in its
-- own pipeline -- via direct instantiation for the one saturate ADD does
-- need, so the final int8 clamp is bit-for-bit the same hardware either
-- module would use.
--
-- UPSAMPLE (nearest-neighbour, factor 2 only -- section 12 limitation 5,
-- 'accel_v2.model.UpsampleOp.factor' default/only supported value; there
-- is no runtime "factor" field in the ISA to read a different value from):
-- iterates one T-byte pixel-plane slice at a time, 'c_tile' outermost,
-- then 'iy', then 'ix' (matching the layout's own nesting), reading it
-- once from 'src0' and replaying the same captured beat into two
-- side-by-side output columns ('2*ix', '2*ix+1' -- contiguous in the
-- tiled layout, so one 2-beat write burst) on each of the two output rows
-- ('2*iy', '2*iy+1'). This never assumes a naive '[H][W][C]' layout: every
-- address is computed from the same
-- '((c_tile*height + y)*width + x)*T' formula
-- 'cnn_accel_model.pack_activation_planes' documents, with 'height'/
-- 'width' substituting the *output* dimensions for the two write
-- addresses. This is deliberately not bandwidth-optimal (one read + two
-- 2-beat writes per input pixel, rather than replaying a whole buffered
-- input row) -- correctness and a small, fixed (one-pixel) buffer were
-- prioritized over throughput; batching whole rows is future work.
entity cnn_accel_elementwise is
  generic (
    -- AXI4-Stream beat width for every channel this entity owns. Must
    -- equal '8 * cnn_accel_constant_activation_plane_channels' (asserted
    -- below) -- see the entity-level "Design simplification" comment.
    g_axi_data_width : positive := 64;
    -- Bound on ADD's per-operand rescale shift amount ('requant_shift'),
    -- the same role 'cnn_accel_bias_requant's own 'g_max_requant_shift'
    -- plays: values above this bound are silently clamped to it (not
    -- reported as an error), the same defensive-not-erroring choice
    -- 'cnn_accel_bias_requant' makes, so the two engines agree on every
    -- input, not just in-range ones.
    g_max_requant_shift : natural := 23;
    -- Defensive upper bound, in bytes, on any single transfer this entity
    -- will ever request (explicit 'xfer_bytes' for COPY/ACT, or the
    -- geometry-computed byte count for ADD/UPSAMPLE's input side).
    -- Exceeding it is reported as 'c_err_bad_geometry' ("oversized").
    g_max_xfer_bytes : positive := 16 * 1024 * 1024
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';

    --# {{}}
    -- Command dispatch: scalar fields lifted from the decoded 'desc_v2_t'
    -- by 'cmd_proc' (mirrors 'cnn_accel_conv_core's 'cfg_*'-port
    -- convention rather than passing the whole record). Sampled only
    -- while idle, alongside 'start'.
    start : in std_ulogic;
    opcode : in std_ulogic_vector(7 downto 0);
    -- desc.in_addr (LOCAL_TENSOR; all four opcodes' first source).
    src0_addr : in unsigned(31 downto 0);
    -- desc.xfer_bytes/src1_addr, ADD's second source address (W15 alias,
    -- see the entity-level comment); ignored by every other opcode.
    src1_addr : in unsigned(31 downto 0);
    -- desc.out_addr (LOCAL_TENSOR; every opcode's destination).
    dst_addr : in unsigned(31 downto 0);
    -- desc.weight_addr, reused as ACT's 256-entry LUT base address (see
    -- the entity-level resolved-ambiguity comment); ignored otherwise.
    lut_addr : in unsigned(31 downto 0);
    -- desc.xfer_bytes, COPY/ACT's byte count; ignored by ADD/UPSAMPLE
    -- (see the entity-level resolved-ambiguity comment on ADD's size).
    xfer_bytes : in unsigned(31 downto 0);
    -- desc.in_width/in_height/in_channels; ADD/UPSAMPLE geometry.
    in_width : in unsigned(15 downto 0);
    in_height : in unsigned(15 downto 0);
    in_channels : in unsigned(15 downto 0);
    -- desc.requant_scale/requant_shift; ADD's single shared (scale,
    -- shift) pair (see the entity-level shared-arithmetic comment).
    requant_scale : in signed(31 downto 0);
    requant_shift : in unsigned(7 downto 0);

    --# {{}}
    -- One-cycle pulse per command, success or failure (see entity-level
    -- comment); 'error'/'error_code' are only meaningful alongside it.
    done : out std_ulogic := '0';
    error : out std_ulogic := '0';
    error_code : out err_code_t := c_err_none;

    --# {{}}
    -- src0: tensor_mem 'r0'. This entity is the read side's consumer.
    src0_req_m2s : out dma_req_m2s_t :=
      (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
    src0_req_s2m : in dma_req_s2m_t;
    s_src0_stream_m2s : in axi_stream_m2s_t;
    s_src0_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;

    --# {{}}
    -- src1: tensor_mem 'r1'. ADD's second source only.
    src1_req_m2s : out dma_req_m2s_t :=
      (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
    src1_req_s2m : in dma_req_s2m_t;
    s_src1_stream_m2s : in axi_stream_m2s_t;
    s_src1_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;

    --# {{}}
    -- dst: tensor_mem 'w1'. This entity is the write side's producer.
    dst_req_m2s : out dma_req_m2s_t :=
      (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
    dst_req_s2m : in dma_req_s2m_t;
    m_dst_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_dst_stream_s2m : in axi_stream_s2m_t;

    --# {{}}
    -- lut: this entity's own LOCAL_WEIGHT-backed 256-entry ACT table read
    -- (see the entity-level comment on why this is a dedicated port).
    lut_req_m2s : out dma_req_m2s_t :=
      (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
    lut_req_s2m : in dma_req_s2m_t;
    s_lut_stream_m2s : in axi_stream_m2s_t;
    s_lut_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init
  );
end entity cnn_accel_elementwise;

architecture a of cnn_accel_elementwise is

  constant c_bytes_per_beat : positive := g_axi_data_width / 8;
  -- 256-entry table: the int8 domain size (2**8), not an ISA-encoded
  -- value -- see spec section 5.2's "256-entry int8->int8 LUT".
  constant c_lut_entries : positive := 256;
  constant c_lut_beats : positive := c_lut_entries / c_bytes_per_beat;

  -- ADD's per-lane product width: int8 (src) * int32 (requant_scale),
  -- exact, no truncation before rounding (same reasoning as
  -- cnn_accel_bias_requant's own 'c_product_width' comment).
  constant c_product_width : positive := 8 + 32;
  constant c_sum_width : positive := c_product_width + 1;

  subtype shift_t is natural range 15 to 15 + g_max_requant_shift;

  type state_t is (
    s_idle,
    s_bad,
    s_lut_req, s_lut_run,
    s_sd_req_src0, s_sd_req_dst, s_sd_run,
    s_add_req_src0, s_add_req_src1, s_add_req_dst, s_add_run,
    s_up_req_src0, s_up_run_src0,
    s_up_req_dst, s_up_run_dst,
    s_up_next,
    s_finish
  );
  signal state_q : state_t := s_idle;

  signal error_code_q : err_code_t := c_err_none;

  -- Latched command fields (sampled once at 'start', see entity-level
  -- comment on why live input ports are not read again afterwards).
  signal opcode_q : std_ulogic_vector(7 downto 0) := (others => '0');
  signal src0_addr_q : unsigned(31 downto 0) := (others => '0');
  signal src1_addr_q : unsigned(31 downto 0) := (others => '0');
  signal dst_addr_q : unsigned(31 downto 0) := (others => '0');
  signal lut_addr_q : unsigned(31 downto 0) := (others => '0');
  signal requant_scale_q : signed(31 downto 0) := (others => '0');
  signal combined_shift_q : shift_t := 15;

  -- Shared "current request" registers (see entity-level comment: only
  -- one '*_req_m2s' is ever asserted at a time, so one pair suffices).
  signal cur_addr_q : unsigned(31 downto 0) := (others => '0');
  signal cur_len_q : unsigned(31 downto 0) := (others => '0');

  -- COPY/ACT/ADD streaming byte count (COPY/ACT: 'xfer_bytes'; ADD: the
  -- geometry-computed padded byte count -- see entity-level comment).
  signal xfer_len_q : unsigned(31 downto 0) := (others => '0');

  -- UPSAMPLE loop state.
  signal n_tiles_q : unsigned(31 downto 0) := (others => '0');
  signal in_w_q : unsigned(31 downto 0) := (others => '0');
  signal in_h_q : unsigned(31 downto 0) := (others => '0');
  signal out_w_q : unsigned(31 downto 0) := (others => '0');
  signal out_h_q : unsigned(31 downto 0) := (others => '0');
  signal c_tile_q : unsigned(31 downto 0) := (others => '0');
  signal iy_q : unsigned(31 downto 0) := (others => '0');
  signal ix_q : unsigned(31 downto 0) := (others => '0');
  -- '0' = top output row ('2*iy'), '1' = bottom output row ('2*iy + 1').
  signal row_phase_q : natural range 0 to 1 := 0;
  -- Which of the 2 replay beats of the current write burst is next.
  signal beat_in_burst_q : natural range 0 to 1 := 0;
  signal pixel_buf_q : std_ulogic_vector(g_axi_data_width - 1 downto 0) := (others => '0');

  -- ACT's 256-entry table, loaded once per command before streaming
  -- begins. Plain registers (not a dual-port BRAM): only 2048 bits, and
  -- ACT needs up to 'c_bytes_per_beat' independent, same-cycle lookups
  -- (one per lane of a beat), which a flat register array gives for free
  -- as combinational muxes -- no read-port arbitration needed.
  type lut_mem_t is array (0 to c_lut_entries - 1) of std_ulogic_vector(7 downto 0);
  signal lut_mem_q : lut_mem_t := (others => (others => '0'));
  signal lut_beat_count_q : natural range 0 to c_lut_beats - 1 := 0;

  -- COPY/ACT streaming data, per lane: passthrough or LUT-mapped byte.
  signal sd_data_next : std_ulogic_vector(g_axi_data_width - 1 downto 0);

  -- ADD streaming data, per lane: the two rescaled operands, summed and
  -- saturated to int8 (see entity-level shared-arithmetic comment).
  signal add_data_next : std_ulogic_vector(g_axi_data_width - 1 downto 0);

  -- dst stream payload mux (one of the three producers, or all-zero while
  -- none apply), separated from 'm_dst_stream_m2s.data' itself because a
  -- conditional expression cannot be nested inside a concatenation.
  signal dst_data_muxed : std_ulogic_vector(g_axi_data_width - 1 downto 0);

  -- Round value/2**shift_amt to the nearest integer, ties towards
  -- +infinity: the exact same guard-bit technique as
  -- cnn_accel_bias_requant.vhd's stage-5 comment ('quot = shift_right(...)',
  -- 'round_up = product_u(shift_amt-1)'), reimplemented unpipelined here
  -- (see entity-level comment on why the whole entity cannot be reused).
  function round_shift_right_signed(value : signed; shift_amt : natural) return signed is
    variable value_u : unsigned(value'range);
    variable quotient : signed(value'range);
  begin
    value_u := unsigned(value);
    quotient := shift_right(value, shift_amt);
    if value_u(shift_amt - 1) = '1' then
      return quotient + 1;
    else
      return quotient;
    end if;
  end function;

  -- 'unsigned "*" unsigned' returns a result of length L'length + R'length
  -- (numeric_std), which grows without bound across a chain of 64-bit
  -- multiplies; every actual quantity here (byte counts/pixel addresses
  -- derived from 16-bit geometry) fits comfortably in 64 bits, so this
  -- helper folds each pairwise product straight back down to 64 bits
  -- (a no-op truncation for every value this entity ever computes) instead
  -- of letting intermediate widths balloon to 128/192/256 bits, which
  -- would fail to elaborate against the 64-bit accumulator variables below.
  function mul64(l, r : unsigned) return unsigned is
  begin
    return resize(l * r, 64);
  end function;

begin

  assert g_axi_data_width = 8 * cnn_accel_constant_activation_plane_channels
    report "cnn_accel_elementwise: g_axi_data_width must equal 8*T " &
           "(cnn_accel_constant_activation_plane_channels) -- see entity-level comment"
    severity failure;

  assert c_lut_entries mod c_bytes_per_beat = 0
    report "cnn_accel_elementwise: c_lut_entries must be a whole multiple of c_bytes_per_beat"
    severity failure;

  assert (15 + g_max_requant_shift) <= (c_product_width - 2)
    report "cnn_accel_elementwise: g_max_requant_shift too large for the ADD product width"
    severity failure;

  ------------------------------------------------------------------------
  -- Combinational outputs.
  ------------------------------------------------------------------------

  done <= '1' when state_q = s_finish or state_q = s_bad else '0';
  error <= '1' when state_q = s_bad else '0';
  error_code <= error_code_q when state_q = s_bad else c_err_none;

  src0_req_m2s.valid <= '1' when
    state_q = s_sd_req_src0 or state_q = s_add_req_src0 or state_q = s_up_req_src0
    else '0';
  src0_req_m2s.req.addr <= cur_addr_q;
  src0_req_m2s.req.length <= cur_len_q;

  src1_req_m2s.valid <= '1' when state_q = s_add_req_src1 else '0';
  src1_req_m2s.req.addr <= cur_addr_q;
  src1_req_m2s.req.length <= cur_len_q;

  dst_req_m2s.valid <= '1' when
    state_q = s_sd_req_dst or state_q = s_add_req_dst or state_q = s_up_req_dst
    else '0';
  dst_req_m2s.req.addr <= cur_addr_q;
  dst_req_m2s.req.length <= cur_len_q;

  lut_req_m2s.valid <= '1' when state_q = s_lut_req else '0';
  lut_req_m2s.req.addr <= cur_addr_q;
  lut_req_m2s.req.length <= cur_len_q;

  -- Stream consumption: ready only in each channel's own run state.
  s_src0_stream_s2m.ready <= '1' when
    (state_q = s_sd_run and m_dst_stream_s2m.ready = '1') or
    (state_q = s_add_run and s_src1_stream_m2s.valid = '1' and m_dst_stream_s2m.ready = '1') or
    state_q = s_up_run_src0
    else '0';

  s_src1_stream_s2m.ready <= '1' when
    state_q = s_add_run and s_src0_stream_m2s.valid = '1' and m_dst_stream_s2m.ready = '1'
    else '0';

  s_lut_stream_s2m.ready <= '1' when state_q = s_lut_run else '0';

  ------------------------------------------------------------------------
  -- COPY/ACT per-lane mapping: passthrough, or through 'lut_mem_q' (ACT).
  ------------------------------------------------------------------------

  sd_lane_gen : for l in 0 to c_bytes_per_beat - 1 generate
    sd_data_next(8 * l + 7 downto 8 * l) <=
      lut_mem_q(to_integer(unsigned(s_src0_stream_m2s.data(8 * l + 7 downto 8 * l))))
        when opcode_q = c_opcode_act else
      s_src0_stream_m2s.data(8 * l + 7 downto 8 * l);
  end generate sd_lane_gen;

  ------------------------------------------------------------------------
  -- ADD per-lane datapath: rescale both operands, sum, saturate to int8.
  -- Always present (combinational), only consumed while state_q = s_add_run
  -- (generate conditions must be locally static -- opcode is a runtime
  -- value, see entity-level comment).
  ------------------------------------------------------------------------

  add_lane_gen : for l in 0 to c_bytes_per_beat - 1 generate
    signal va, vb : signed(7 downto 0);
    signal prod_a, prod_b : signed(c_product_width - 1 downto 0);
    signal ra, rb : signed(c_product_width - 1 downto 0);
    signal sum_ext : signed(c_sum_width - 1 downto 0);
    signal sat_byte : signed(7 downto 0);
  begin
    va <= signed(s_src0_stream_m2s.data(8 * l + 7 downto 8 * l));
    vb <= signed(s_src1_stream_m2s.data(8 * l + 7 downto 8 * l));

    prod_a <= va * requant_scale_q;
    prod_b <= vb * requant_scale_q;

    ra <= round_shift_right_signed(prod_a, combined_shift_q);
    rb <= round_shift_right_signed(prod_b, combined_shift_q);

    sum_ext <= resize(ra, c_sum_width) + resize(rb, c_sum_width);

    saturate_inst : entity math.saturate_signed
      generic map (
        input_width => c_sum_width,
        result_width => 8,
        enable_output_register => false
      )
      port map (
        clk => clk,
        input_valid => '1',
        input_value => sum_ext,
        result_valid => open,
        result_value => sat_byte,
        result_is_saturated => open
      );

    add_data_next(8 * l + 7 downto 8 * l) <= std_ulogic_vector(sat_byte);
  end generate add_lane_gen;

  ------------------------------------------------------------------------
  -- dst stream mux: one of the three producers, selected by 'state_q'.
  ------------------------------------------------------------------------

  -- A conditional expression ('... when ... else ...') cannot be nested as
  -- an operand of '&' (GHDL rejects it, LRM conditional expressions are
  -- only legal as a whole waveform/expression, not a sub-expression), so
  -- the payload mux is a separate signal, concatenated with zero-padding
  -- below rather than inline.
  dst_data_muxed <=
    sd_data_next when state_q = s_sd_run else
    add_data_next when state_q = s_add_run else
    pixel_buf_q when state_q = s_up_run_dst else
    (g_axi_data_width - 1 downto 0 => '0');

  m_dst_stream_m2s.data <= (axi_stream_data_sz - 1 downto g_axi_data_width => '0') &
    dst_data_muxed;

  m_dst_stream_m2s.valid <=
    s_src0_stream_m2s.valid when state_q = s_sd_run else
    (s_src0_stream_m2s.valid and s_src1_stream_m2s.valid) when state_q = s_add_run else
    '1' when state_q = s_up_run_dst else
    '0';

  m_dst_stream_m2s.last <=
    s_src0_stream_m2s.last when state_q = s_sd_run else
    -- src0/src1/dst were all requested with the same length ('xfer_len_q'),
    -- so their 'last' beats coincide; see entity-level comment.
    s_src0_stream_m2s.last when state_q = s_add_run else
    '1' when (state_q = s_up_run_dst and beat_in_burst_q = 1) else
    '0';

  ------------------------------------------------------------------------
  -- FSM.
  ------------------------------------------------------------------------

  fsm : process(clk)
    variable in_w64, in_h64, in_c64, n_tiles64, total64, out_w64, out_h64 : unsigned(63 downto 0);
    variable pixel_idx64, out_idx64, addr64, tmp64 : unsigned(63 downto 0);
    variable shift_raw : natural;
    variable bad : boolean;
    -- 's_up_next' scratch: the loop counters for the *next* pixel this
    -- state advances to (computed here so the following 's_up_req_src0'
    -- request's address can be latched in the same state, rather than
    -- split across a second process -- see entity-level comment history).
    variable next_ix_v, next_iy_v, next_tile_v : unsigned(31 downto 0);
    variable last_pixel : boolean;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        state_q <= s_idle;
        lut_beat_count_q <= 0;
        row_phase_q <= 0;
        beat_in_burst_q <= 0;
        error_code_q <= c_err_none;
      else
        case state_q is

          ------------------------------------------------------------------
          when s_idle =>
            if start = '1' then
              opcode_q <= opcode;
              src0_addr_q <= src0_addr;
              src1_addr_q <= src1_addr;
              dst_addr_q <= dst_addr;
              lut_addr_q <= lut_addr;
              requant_scale_q <= requant_scale;

              shift_raw := to_integer(requant_shift);
              if shift_raw > g_max_requant_shift then
                combined_shift_q <= 15 + g_max_requant_shift;
              else
                combined_shift_q <= 15 + shift_raw;
              end if;

              if opcode = c_opcode_copy or opcode = c_opcode_act then
                bad := xfer_bytes = 0 or
                       (xfer_bytes mod to_unsigned(c_bytes_per_beat, 32)) /= 0 or
                       xfer_bytes > to_unsigned(g_max_xfer_bytes, 32);
                if bad then
                  error_code_q <= c_err_bad_geometry;
                  state_q <= s_bad;
                else
                  xfer_len_q <= xfer_bytes;
                  if opcode = c_opcode_act then
                    cur_addr_q <= lut_addr;
                    cur_len_q <= to_unsigned(c_lut_entries, 32);
                    lut_beat_count_q <= 0;
                    state_q <= s_lut_req;
                  else
                    cur_addr_q <= src0_addr;
                    cur_len_q <= xfer_bytes;
                    state_q <= s_sd_req_src0;
                  end if;
                end if;

              elsif opcode = c_opcode_add then
                in_w64 := resize(in_width, 64);
                in_h64 := resize(in_height, 64);
                in_c64 := resize(in_channels, 64);
                n_tiles64 := (in_c64 + to_unsigned(c_bytes_per_beat, 64) - 1) /
                             to_unsigned(c_bytes_per_beat, 64);
                total64 := mul64(mul64(mul64(n_tiles64, in_h64), in_w64),
                                  to_unsigned(c_bytes_per_beat, 64));
                bad := in_w64 = 0 or in_h64 = 0 or in_c64 = 0 or
                       total64 = 0 or total64 > to_unsigned(g_max_xfer_bytes, 64);
                if bad then
                  error_code_q <= c_err_bad_geometry;
                  state_q <= s_bad;
                else
                  xfer_len_q <= resize(total64, 32);
                  cur_addr_q <= src0_addr;
                  cur_len_q <= resize(total64, 32);
                  state_q <= s_add_req_src0;
                end if;

              elsif opcode = c_opcode_upsample then
                in_w64 := resize(in_width, 64);
                in_h64 := resize(in_height, 64);
                in_c64 := resize(in_channels, 64);
                n_tiles64 := (in_c64 + to_unsigned(c_bytes_per_beat, 64) - 1) /
                             to_unsigned(c_bytes_per_beat, 64);
                -- 'unsigned "*" natural' (numeric_std A.17) converts the
                -- natural to an unsigned of L'length bits before
                -- multiplying, so a bare 'in_w64 * 2' produces a
                -- 64+64 = 128-bit result -- a length mismatch against
                -- these 64-bit variables that GHDL only catches at
                -- runtime (bound check failure), not at analysis time.
                -- Route through 'mul64' like every other product in this
                -- process, for the same reason its own comment gives.
                out_w64 := mul64(in_w64, to_unsigned(2, 64));
                out_h64 := mul64(in_h64, to_unsigned(2, 64));
                total64 := mul64(mul64(mul64(n_tiles64, out_h64), out_w64),
                                  to_unsigned(c_bytes_per_beat, 64));
                bad := in_w64 = 0 or in_h64 = 0 or in_c64 = 0 or
                       total64 = 0 or total64 > to_unsigned(g_max_xfer_bytes, 64);
                if bad then
                  error_code_q <= c_err_bad_geometry;
                  state_q <= s_bad;
                else
                  n_tiles_q <= resize(n_tiles64, 32);
                  in_w_q <= resize(in_w64, 32);
                  in_h_q <= resize(in_h64, 32);
                  out_w_q <= resize(out_w64, 32);
                  out_h_q <= resize(out_h64, 32);
                  c_tile_q <= (others => '0');
                  iy_q <= (others => '0');
                  ix_q <= (others => '0');
                  row_phase_q <= 0;

                  addr64 := resize(src0_addr, 64);
                  cur_addr_q <= resize(addr64, 32);
                  cur_len_q <= to_unsigned(c_bytes_per_beat, 32);
                  state_q <= s_up_req_src0;
                end if;

              else
                error_code_q <= c_err_unsupported_op;
                state_q <= s_bad;
              end if;
            end if;

          ------------------------------------------------------------------
          when s_bad =>
            state_q <= s_idle;

          when s_finish =>
            state_q <= s_idle;

          ------------------------------------------------------------------
          -- ACT: preload the 256-entry LUT before streaming.
          ------------------------------------------------------------------
          when s_lut_req =>
            if lut_req_s2m.ready = '1' then
              state_q <= s_lut_run;
            end if;

          when s_lut_run =>
            if s_lut_stream_m2s.valid = '1' then
              for lane in 0 to c_bytes_per_beat - 1 loop
                lut_mem_q(lut_beat_count_q * c_bytes_per_beat + lane) <=
                  s_lut_stream_m2s.data(8 * lane + 7 downto 8 * lane);
              end loop;
              if s_lut_stream_m2s.last = '1' then
                cur_addr_q <= src0_addr_q;
                cur_len_q <= xfer_len_q;
                state_q <= s_sd_req_src0;
              else
                lut_beat_count_q <= lut_beat_count_q + 1;
              end if;
            end if;

          ------------------------------------------------------------------
          -- COPY / ACT: stream src0 -> (LUT or passthrough) -> dst.
          ------------------------------------------------------------------
          when s_sd_req_src0 =>
            if src0_req_s2m.ready = '1' then
              cur_addr_q <= dst_addr_q;
              cur_len_q <= xfer_len_q;
              state_q <= s_sd_req_dst;
            end if;

          when s_sd_req_dst =>
            if dst_req_s2m.ready = '1' then
              state_q <= s_sd_run;
            end if;

          when s_sd_run =>
            if s_src0_stream_m2s.valid = '1' and m_dst_stream_s2m.ready = '1' and
               s_src0_stream_m2s.last = '1' then
              state_q <= s_finish;
            end if;

          ------------------------------------------------------------------
          -- ADD: stream src0 & src1 (joined) -> rescale/sum/saturate -> dst.
          ------------------------------------------------------------------
          when s_add_req_src0 =>
            if src0_req_s2m.ready = '1' then
              cur_addr_q <= src1_addr_q;
              cur_len_q <= xfer_len_q;
              state_q <= s_add_req_src1;
            end if;

          when s_add_req_src1 =>
            if src1_req_s2m.ready = '1' then
              cur_addr_q <= dst_addr_q;
              cur_len_q <= xfer_len_q;
              state_q <= s_add_req_dst;
            end if;

          when s_add_req_dst =>
            if dst_req_s2m.ready = '1' then
              state_q <= s_add_run;
            end if;

          when s_add_run =>
            if s_src0_stream_m2s.valid = '1' and s_src1_stream_m2s.valid = '1' and
               m_dst_stream_s2m.ready = '1' and s_src0_stream_m2s.last = '1' then
              state_q <= s_finish;
            end if;

          ------------------------------------------------------------------
          -- UPSAMPLE: one T-byte pixel-plane per iteration.
          ------------------------------------------------------------------
          when s_up_req_src0 =>
            if src0_req_s2m.ready = '1' then
              state_q <= s_up_run_src0;
            end if;

          when s_up_run_src0 =>
            -- Exactly one beat was requested; the first (and only) valid
            -- beat is always the last.
            if s_src0_stream_m2s.valid = '1' then
              pixel_buf_q <= s_src0_stream_m2s.data(g_axi_data_width - 1 downto 0);

              -- out_pixel_index = (c_tile*out_h + (2*iy + row_phase))*out_w + 2*ix
              tmp64 := mul64(c_tile_q, out_h_q) + resize(2 * iy_q, 64) +
                       to_unsigned(row_phase_q, 64);
              out_idx64 := mul64(tmp64, out_w_q) + resize(2 * ix_q, 64);
              addr64 := resize(dst_addr_q, 64) + mul64(out_idx64, to_unsigned(c_bytes_per_beat, 64));
              cur_addr_q <= resize(addr64, 32);
              cur_len_q <= to_unsigned(2 * c_bytes_per_beat, 32);
              beat_in_burst_q <= 0;
              state_q <= s_up_req_dst;
            end if;

          when s_up_req_dst =>
            if dst_req_s2m.ready = '1' then
              state_q <= s_up_run_dst;
            end if;

          when s_up_run_dst =>
            if m_dst_stream_s2m.ready = '1' then
              if beat_in_burst_q = 1 then
                if row_phase_q = 0 then
                  row_phase_q <= 1;

                  tmp64 := mul64(c_tile_q, out_h_q) + resize(2 * iy_q, 64) + to_unsigned(1, 64);
                  out_idx64 := mul64(tmp64, out_w_q) + resize(2 * ix_q, 64);
                  addr64 := resize(dst_addr_q, 64) + mul64(out_idx64, to_unsigned(c_bytes_per_beat, 64));
                  cur_addr_q <= resize(addr64, 32);
                  cur_len_q <= to_unsigned(2 * c_bytes_per_beat, 32);
                  beat_in_burst_q <= 0;
                  state_q <= s_up_req_dst;
                else
                  row_phase_q <= 0;
                  state_q <= s_up_next;
                end if;
              else
                beat_in_burst_q <= 1;
              end if;
            end if;

          when s_up_next =>
            -- Compute the *next* pixel's loop counters here (rather than
            -- advancing them and re-deriving the address in a second,
            -- same-edge process): a second process triggered off the same
            -- clock cannot observe this process's own not-yet-committed
            -- signal writes, so any address derived from 'ix_q'/'iy_q'/
            -- 'c_tile_q' outside of this process would only ever see the
            -- *pre-advance* counters, one pixel stale. Doing both the
            -- advance and the address computation in the same state, from
            -- the same local variables, avoids that hazard entirely.
            last_pixel := ix_q >= in_w_q - 1 and iy_q >= in_h_q - 1 and
                          c_tile_q >= n_tiles_q - 1;
            if last_pixel then
              state_q <= s_finish;
            else
              if ix_q < in_w_q - 1 then
                next_ix_v := ix_q + 1;
                next_iy_v := iy_q;
                next_tile_v := c_tile_q;
              elsif iy_q < in_h_q - 1 then
                next_ix_v := (others => '0');
                next_iy_v := iy_q + 1;
                next_tile_v := c_tile_q;
              else
                next_ix_v := (others => '0');
                next_iy_v := (others => '0');
                next_tile_v := c_tile_q + 1;
              end if;

              ix_q <= next_ix_v;
              iy_q <= next_iy_v;
              c_tile_q <= next_tile_v;

              -- pixel_idx = (c_tile*in_h + iy)*in_w + ix, for the pixel
              -- about to be latched above (the *next* counters, not the
              -- current ones).
              tmp64 := mul64(next_tile_v, in_h_q) + resize(next_iy_v, 64);
              pixel_idx64 := mul64(tmp64, in_w_q) + resize(next_ix_v, 64);
              addr64 := resize(src0_addr_q, 64) +
                        mul64(pixel_idx64, to_unsigned(c_bytes_per_beat, 64));
              cur_addr_q <= resize(addr64, 32);
              cur_len_q <= to_unsigned(c_bytes_per_beat, 32);
              state_q <= s_up_req_src0;
            end if;

        end case;
      end if;
    end if;
  end process;

end architecture a;
