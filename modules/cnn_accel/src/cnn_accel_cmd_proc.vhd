library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;
use cnn_accel.cnn_accel_v2_pkg.all;
use cnn_accel.cnn_accel_isa_pkg.c_instr_word_bytes;
use cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_activation_plane_channels;
use cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_scale_table_entry_bytes;
use cnn_accel.cnn_accel_regs_pkg.cnn_accel_constant_accum_width;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

-- ISA v2.0 command processor (doc/cnn_accel_top_v2_arch.md section 2:
-- "decode / validate / dispatch / OT loop / retire"). Replaces rev 1's
-- never-implemented 'cnn_accel_sequencer' + 'cnn_accel_layer_ctrl' pair
-- with a single FSM that owns the program counter, descriptor validation
-- (section 9), operand-space resolution (section 3), engine dispatch, the
-- output-channel-tile loop (section 5.4), retirement and every
-- performance counter (section 8).
--
-- Deliberately free of neural-network semantics. This entity knows about
-- commands, storage references, byte counts and engines; it does not know
-- what a network, a layer or a residual is. Everything it computes is
-- either an ISA field, a byte address, a byte count, or a handshake.
--
-- Operand routing (why this module is also the interconnect). The engines
-- and DMAs all speak the same '(addr, length)' 'dma_req_m2s_t' + AXI4-
-- Stream idiom (section 4), and a command's space tags decide, per
-- operand, which physical port that idiom is bound to for the duration of
-- the command. That binding is exactly "operand-space resolution", so it
-- lives here rather than in 'cnn_accel_top', which stays purely
-- structural. Because only one command is ever in flight (section 12
-- limitation 1), every mux below is a plain command-scoped selection with
-- no arbitration and no fairness question.
--
--   operand          | DDR                     | LOCAL_TENSOR | LOCAL_WEIGHT
--   -----------------+-------------------------+--------------+--------------
--   src0 / activation| 'load' axi_read_dma     | tensor_mem r0| illegal
--   src1 (ADD only)  | 'wgt' axi_read_dma      | tensor_mem r1| illegal
--   dst              | ofmap_dma               | tensor_mem w1| LOADW only
--   weight/bias/scale| 'wgt' axi_read_dma      | tensor_mem r1| resident
--   LUT (ACT)        | 'wgt' axi_read_dma      | tensor_mem r1| resident
--
-- 'src1' and the weight/LUT operand share the 'wgt' read DMA and
-- 'tensor_mem r1' because no opcode ever uses both: 'ADD' is the only
-- opcode with a second source and it has neither weights nor a LUT.
-- 'tensor_mem w0' is the DDR->LOCAL landing channel ('LOAD'), 'w1' the
-- engine-output channel, exactly as section 4's channel table assigns
-- them.
--
-- Residency (section 10). Nothing in this module ever moves a tensor to
-- DDR on its own initiative: a DDR write request is issued if and only if
-- the in-flight command's 'space_dst' is 'DDR'. A compute command with
-- 'space_src0 = space_dst = LOCAL_TENSOR' therefore issues no AXI
-- transaction at all beyond its own descriptor fetch, which is what makes
-- R1/R2/R4 hold by construction rather than by test.
--
-- Ifmap plane de-interleaving (a layout mismatch this module has to
-- absorb). Activations live in DDR *and* in the scratchpad in the section
-- 3 PLANES layout, '[c_tile][y][x][t]' -- channel tile outermost.
-- 'cnn_accel_window_gen' (inside 'cnn_accel_conv_core') ingests one
-- '(col, tile)' cell per beat in '[y][x][t]' order -- channel tile
-- *innermost*. The two orders coincide only while 'T = 1'
-- ('in_channels <= 8'). Rev-1's 'cnn_accel_layer_ctrl' proposal noted the
-- problem in passing ("planes are streamed plane-major only when
-- in_channels <= T") but never resolved it, and was never implemented, so
-- it has never been exercised. It is resolved here, explicitly, and in
-- the only place that can resolve it without touching a reused engine:
-- for 'T > 1' the ifmap is fetched one '(row, tile)' strip at a time into
-- a single-row transpose buffer sized by the same 'g_max_row_tile_words'
-- bound 'cnn_accel_window_gen' already imposes on 'in_width * T', and
-- drained in '[x][t]' order. For 'T = 1' the buffer is bypassed and the
-- whole tensor is one request, so the common case pays nothing.
--
-- No-hang guarantee (section 9). Every state that waits on something
-- external -- descriptor fetch, a DMA request handshake, a DMA
-- completion pulse, an engine 'done' -- is covered by the same
-- free-running watchdog: 'watchdog_q' is reloaded on every state change
-- and on every observable forward progress event, and its expiry is an
-- unconditional transition to 'st_error' with 'c_err_timeout'.
-- 'soft_reset_pulse' is sampled in every state and aborts to 'st_idle'.
-- The FSM therefore always reaches 'st_done' or 'st_error' in bounded
-- time regardless of what any engine, DMA or AXI slave does.
entity cnn_accel_cmd_proc is
  generic (
    -- Output-channel parallelism; the OT loop's tile size (section 5.4).
    g_pe_rows : positive;
    -- Input-channel/MAC parallelism; weight-image lane count per row.
    g_pe_cols : positive;
    -- Input channels per window_gen beat ("Ct").
    g_tile_channels : positive;
    -- Upper bound on kernel_h/kernel_w, for 'c_err_bad_geometry'.
    g_max_kernel_size : positive;
    -- Upper bound on pool_kernel_h/pool_kernel_w, for
    -- 'c_err_bad_geometry'. Separate from (and larger than)
    -- 'g_max_kernel_size' -- see cnn_accel_top's own generic comment.
    g_max_pool_kernel_size : positive;
    -- Upper bound on 'in_width * ceil(in_channels/g_tile_channels)', for
    -- 'c_err_bad_geometry' and for sizing the transpose buffer above.
    g_max_row_tile_words : positive;
    -- Size of 'cnn_accel_tensor_mem', bytes; bound for 'c_err_local_range'.
    g_tensor_bytes : positive;
    -- Exclusive upper bound on any DDR byte address ('g_ddr_limit',
    -- section 6); bound for 'c_err_ddr_range'.
    g_ddr_limit : positive := 16#0020_0000#;
    -- Cycles any single wait may take before 'c_err_timeout' (section 9).
    g_watchdog_cycles : positive := 1_000_000
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';

    --# {{}}
    -- Run control from 'cnn_accel_csr'.
    start : in std_ulogic;
    program_base_addr : in std_ulogic_vector(31 downto 0);
    soft_reset_pulse : in std_ulogic;
    seq_done : out std_ulogic := '0';
    seq_error : out std_ulogic := '0';
    err_code : out std_ulogic_vector(3 downto 0) := (others => '0');
    err_pc : out std_ulogic_vector(31 downto 0) := (others => '0');
    counters : out csr_counters_t := csr_counters_init;

    --# {{}}
    -- Per-cycle external-traffic increments from 'cnn_accel_axi_mux'.
    axi_rd_bytes : in unsigned(7 downto 0);
    axi_wr_bytes : in unsigned(7 downto 0);

    --# {{}}
    -- 'cnn_accel_cmd_fetch'.
    fetch_start : out std_ulogic := '0';
    fetch_addr : out unsigned(31 downto 0) := (others => '0');
    fetch_desc : in desc_v2_t;
    fetch_pc : in unsigned(31 downto 0);
    fetch_desc_valid : in std_ulogic;
    fetch_desc_ready : out std_ulogic := '0';
    fetch_error : in std_ulogic;
    fetch_error_code : in err_code_t;

    --# {{}}
    -- 'load' DDR read DMA: activations and 'LOAD'.
    load_req_m2s : out dma_req_m2s_t;
    load_req_s2m : in dma_req_s2m_t;
    load_dma_done : in std_ulogic;
    load_resp_error : in std_ulogic;
    s_load_stream_m2s : in axi_stream_m2s_t;
    s_load_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;

    --# {{}}
    -- 'wgt' DDR read DMA: weight/bias/scale images, ACT LUT, ADD src1.
    wgt_req_m2s : out dma_req_m2s_t;
    wgt_req_s2m : in dma_req_s2m_t;
    wgt_dma_done : in std_ulogic;
    wgt_resp_error : in std_ulogic;
    s_wgt_stream_m2s : in axi_stream_m2s_t;
    s_wgt_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;

    --# {{}}
    -- 'cnn_accel_ofmap_dma': every DDR write the IP ever performs.
    store_req_m2s : out dma_req_m2s_t;
    store_req_s2m : in dma_req_s2m_t;
    store_dma_done : in std_ulogic;
    store_resp_error : in std_ulogic;
    m_store_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_store_stream_s2m : in axi_stream_s2m_t;

    --# {{}}
    -- 'cnn_accel_tensor_mem' write channel 0 (DDR -> LOCAL landing).
    tm_w0_req_m2s : out dma_req_m2s_t;
    tm_w0_req_s2m : in dma_req_s2m_t;
    m_tm_w0_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_tm_w0_s2m : in axi_stream_s2m_t;
    tm_w0_done : in std_ulogic;

    --# {{}}
    -- 'cnn_accel_tensor_mem' write channel 1 (engine output).
    tm_w1_req_m2s : out dma_req_m2s_t;
    tm_w1_req_s2m : in dma_req_s2m_t;
    m_tm_w1_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_tm_w1_s2m : in axi_stream_s2m_t;
    tm_w1_done : in std_ulogic;

    --# {{}}
    -- 'cnn_accel_tensor_mem' read channel 0 (engine input A / store source).
    tm_r0_req_m2s : out dma_req_m2s_t;
    tm_r0_req_s2m : in dma_req_s2m_t;
    s_tm_r0_m2s : in axi_stream_m2s_t;
    s_tm_r0_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;
    tm_r0_done : in std_ulogic;

    --# {{}}
    -- 'cnn_accel_tensor_mem' read channel 1 (engine input B / weights / LUT).
    tm_r1_req_m2s : out dma_req_m2s_t;
    tm_r1_req_s2m : in dma_req_s2m_t;
    s_tm_r1_m2s : in axi_stream_m2s_t;
    s_tm_r1_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;
    tm_r1_done : in std_ulogic;

    --# {{}}
    -- 'cnn_accel_conv_core' configuration and control.
    conv_cfg_kernel_h : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_kernel_w : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_stride_h : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_stride_w : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_pad_top : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_pad_bottom : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_pad_left : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_pad_right : out std_ulogic_vector(7 downto 0) := (others => '0');
    -- ISA v2.1 'pad_value' for convolution: the int8 value a padded tap
    -- takes -- the input tensor's quantization zero-point, not 0. Driven
    -- straight from the descriptor and NOT gated on FLAG_PAD_EN, exactly
    -- like 'pool_cfg_pad_value' below: with the flag clear the four pad
    -- counts above are already zero, so no tap is ever padded and the
    -- fill value cannot be observed.
    conv_cfg_pad_value : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_in_width : out std_ulogic_vector(15 downto 0) := (others => '0');
    conv_cfg_in_height : out std_ulogic_vector(15 downto 0) := (others => '0');
    conv_cfg_in_channels : out std_ulogic_vector(15 downto 0) := (others => '0');
    conv_cfg_bias_en : out std_ulogic := '0';
    conv_cfg_requant_en : out std_ulogic := '0';
    conv_cfg_relu_en : out std_ulogic := '0';
    conv_cfg_requant_scale : out std_ulogic_vector(31 downto 0) := (others => '0');
    conv_cfg_requant_shift : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_output_offset : out std_ulogic_vector(15 downto 0) := (others => '0');
    conv_cfg_clamp_en : out std_ulogic := '0';
    conv_cfg_clamp_min : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_clamp_max : out std_ulogic_vector(7 downto 0) := (others => '0');
    conv_cfg_per_channel_en : out std_ulogic := '0';
    -- Output frame dimensions for the command being started:
    -- '(in_dim + pad_lo + pad_hi - kernel) / stride + 1' per axis, from
    -- the 'st_div_w'/'st_div_h' restoring divider, i.e. the same
    -- 'out_w_q'/'out_h_q' this module already uses to size the output
    -- plane. Both window generators (conv and pool) latch these at their
    -- 'start' rather than dividing again in one combinational cone off
    -- 'desc_q', which is what made 'out_height_q' a 135-logic-level
    -- endpoint. They are shared across the two engines because only one
    -- engine is ever started for a given descriptor, and the divider is
    -- fed from the conv or the pool kernel/stride fields according to
    -- that same class decision.
    geom_out_width : out std_ulogic_vector(15 downto 0) := (others => '0');
    geom_out_height : out std_ulogic_vector(15 downto 0) := (others => '0');
    conv_start : out std_ulogic := '0';
    conv_done : in std_ulogic;
    m_conv_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_conv_stream_s2m : in axi_stream_s2m_t;
    conv_fill_start : out std_ulogic := '0';
    conv_fill_is_bias : out std_ulogic := '0';
    conv_fill_is_scale : out std_ulogic := '0';
    m_conv_weight_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_conv_weight_s2m : in axi_stream_s2m_t;
    s_conv_out_m2s : in axi_stream_m2s_t;
    s_conv_out_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;

    --# {{}}
    -- Pooling path: the top level's dedicated 'cnn_accel_window_gen'
    -- instance plus its lane-parallel 'cnn_accel_pool' bank (see
    -- 'cnn_accel_top'). One activation plane is processed per pass, so
    -- the window generator always runs with 'T = 1'.
    pool_cfg_kernel_h : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_kernel_w : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_stride_h : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_stride_w : out std_ulogic_vector(7 downto 0) := (others => '0');
    -- ISA v2.1 pooling padding. The four counts are already gated on
    -- FLAG_PAD_EN here (zero when the flag is clear), exactly as the
    -- conv path's are, so the window generator never has to know about
    -- the flag. 'pool_cfg_pad_value' is the int8 value a padded tap
    -- takes -- the tensor's zero-point, not 0.
    pool_cfg_pad_top : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_pad_bottom : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_pad_left : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_pad_right : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_pad_value : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_in_width : out std_ulogic_vector(15 downto 0) := (others => '0');
    pool_cfg_in_height : out std_ulogic_vector(15 downto 0) := (others => '0');
    pool_cfg_opcode : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_cfg_requant_scale : out std_ulogic_vector(31 downto 0) := (others => '0');
    pool_cfg_requant_shift : out std_ulogic_vector(7 downto 0) := (others => '0');
    pool_start : out std_ulogic := '0';
    pool_done : in std_ulogic;
    m_pool_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_pool_stream_s2m : in axi_stream_s2m_t;
    s_pool_out_m2s : in axi_stream_m2s_t;
    s_pool_out_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;

    --# {{}}
    -- 'cnn_accel_elementwise' (ADD / UPSAMPLE / COPY / ACT).
    ew_start : out std_ulogic := '0';
    ew_opcode : out std_ulogic_vector(7 downto 0) := (others => '0');
    ew_src0_addr : out unsigned(31 downto 0) := (others => '0');
    ew_src1_addr : out unsigned(31 downto 0) := (others => '0');
    ew_dst_addr : out unsigned(31 downto 0) := (others => '0');
    ew_lut_addr : out unsigned(31 downto 0) := (others => '0');
    ew_xfer_bytes : out unsigned(31 downto 0) := (others => '0');
    ew_in_width : out unsigned(15 downto 0) := (others => '0');
    ew_in_height : out unsigned(15 downto 0) := (others => '0');
    ew_in_channels : out unsigned(15 downto 0) := (others => '0');
    ew_requant_scale : out signed(31 downto 0) := (others => '0');
    ew_requant_shift : out unsigned(7 downto 0) := (others => '0');
    ew_done : in std_ulogic;
    ew_error : in std_ulogic;
    ew_error_code : in err_code_t;
    -- The engine's own four request/stream ports, muxed onto the physical
    -- ports above according to the command's space tags.
    ew_src0_req_m2s : in dma_req_m2s_t;
    ew_src0_req_s2m : out dma_req_s2m_t;
    m_ew_src0_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_ew_src0_stream_s2m : in axi_stream_s2m_t;
    ew_src1_req_m2s : in dma_req_m2s_t;
    ew_src1_req_s2m : out dma_req_s2m_t;
    m_ew_src1_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_ew_src1_stream_s2m : in axi_stream_s2m_t;
    ew_dst_req_m2s : in dma_req_m2s_t;
    ew_dst_req_s2m : out dma_req_s2m_t;
    s_ew_dst_stream_m2s : in axi_stream_m2s_t;
    s_ew_dst_stream_s2m : out axi_stream_s2m_t := axi_stream_s2m_init;
    ew_lut_req_m2s : in dma_req_m2s_t;
    ew_lut_req_s2m : out dma_req_s2m_t;
    m_ew_lut_stream_m2s : out axi_stream_m2s_t := axi_stream_m2s_init;
    m_ew_lut_stream_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_cmd_proc;

architecture a of cnn_accel_cmd_proc is

  ------------------------------------------------------------------------
  -- Layout constants. Every one of them is either a generic or a
  -- generated constant; no instruction-word or memory-layout number is
  -- written as a literal in this file.
  ------------------------------------------------------------------------

  -- Channels per activation plane word ("T" of decision S6). The whole
  -- PLANES layout -- and therefore every byte count computed below --
  -- follows from this and from the fact that one activation is one byte.
  constant c_plane_channels : positive := cnn_accel_constant_activation_plane_channels;
  -- Bytes in one activation plane word; also the AXI/stream beat size the
  -- scratchpad and both DMAs work in.
  constant c_word_bytes : positive := c_plane_channels;
  -- log2 of the above, for the alignment and byte<->beat conversions.
  constant c_word_shift : positive := 3;

  constant c_bias_entry_bytes : positive := cnn_accel_constant_accum_width / 8;
  constant c_scale_entry_bytes : positive := cnn_accel_constant_scale_table_entry_bytes;

  constant c_dma_req_init : dma_req_m2s_t :=
    (valid => '0', req => (addr => (others => '0'), length => (others => '0')));

  ------------------------------------------------------------------------
  -- Command classification. Purely a decode of the opcode field; kept as
  -- an enumeration so every downstream 'case' is exhaustive and a new
  -- opcode cannot be silently forgotten.
  ------------------------------------------------------------------------

  type cmd_class_t is (
    cls_halt,      -- terminate the program
    cls_conv,      -- CONV2D / FC: conv_core, OT loop
    cls_pool,      -- POOL_MAX / POOL_AVG: window_gen + pool bank
    cls_xfer,      -- LOAD / STORE / LOADW: pure move, no engine
    cls_elem,      -- ADD / UPSAMPLE / COPY / ACT: elementwise
    cls_bad        -- anything else, including DWCONV2D
  );

  type state_t is (
    st_idle,
    st_fetch, st_fetch_wait,
    st_precheck, st_validate, st_validate2,
    st_geom_mul, st_div_w, st_div_h, st_geom_out, st_geom_out2,
    st_range_sum, st_range, st_range_dst,
    st_wgt_setup, st_wgt_req, st_wgt_run,
    st_pass_setup, st_pass_req_dst, st_pass_run, st_pass_drain,
    st_xfer_req_src, st_xfer_req_dst, st_xfer_run,
    st_elem_run,
    st_retire, st_error, st_done
  );

  signal state : state_t := st_idle;

  ------------------------------------------------------------------------
  -- Latched command.
  ------------------------------------------------------------------------

  signal desc_q : desc_v2_t := desc_v2_init;
  signal pc_q : unsigned(31 downto 0) := (others => '0');
  signal cls_q : cmd_class_t := cls_halt;

  ------------------------------------------------------------------------
  -- Pre-decoded descriptor predicates ('st_precheck').
  --
  -- 'st_validate' used to test the raw descriptor: one 8-bit opcode
  -- compared against ~15 literals scattered through the reserved-field,
  -- space-tag, alignment and geometry rules, each of those rules itself a
  -- wide '/= 0' or '<=' over a 16/32-bit field, and all of it collapsing
  -- into one priority chain that ends on 'pending_err_q's clock enable and
  -- 'state'. Post-route that was 19 logic levels with 8 CARRY4 and
  -- -1.218 ns -- the second failing family in the design, and the exact
  -- shape 'shared/TimingAndResources.md' Fundamentals, "Control structure"
  -- names: "one wide field (an opcode, a mode) compared against a dozen-plus
  -- literals scattered across a controller's validation and dispatch logic".
  --
  -- The fix it prescribes is to decode the whole set into one-bit flags
  -- once, registered, and have every site test its flag. 'st_precheck' is
  -- that cycle: every wide comparison the validation rules need is reduced
  -- to a single registered bit here, so 'st_validate' is a chain of 1-bit
  -- terms plus the (3-bit) space tags. It costs exactly one cycle per
  -- instruction, against commands that run for thousands.
  signal op_is_add_q : std_ulogic := '0';
  signal op_is_act_q : std_ulogic := '0';
  signal op_is_loadw_q : std_ulogic := '0';
  signal op_is_copy_q : std_ulogic := '0';
  signal op_is_v12_q : std_ulogic := '0';
  signal op_is_upsample_q : std_ulogic := '0';

  -- Source/destination extents ('base + length'), computed in
  -- 'st_range_sum' and only COMPARED in 'st_range'/'st_range_dst'.
  --
  -- The range checks used to select the operand length off the raw opcode
  -- and then do 'resize(addr, 33) + len > limit' in the same cycle as the
  -- error-priority chain that consumes the result. Post-route that was the
  -- design's worst path (-0.697 ns, 15-16 levels: one opcode-decode LUT,
  -- the length mux, SIX chained CARRY4 for the 33-bit add-and-compare, then
  -- five LUT levels of priority chain into 'pending_err_q's clock enable
  -- and 'state'). 'shared/TimingAndResources.md', Fundamentals: "Split
  -- adders wider than the budget allows across stages", and section 2 --
  -- this is per-command work, so a cycle is free and a logic level is not.
  signal src_end_q : unsigned(32 downto 0) := (others => '0');
  signal dst_end_q : unsigned(32 downto 0) := (others => '0');

  -- 'in_width + pad_left + pad_right' and 'in_height + pad_top +
  -- pad_bottom', gated by 'FLAG_PAD_EN', precomputed in 'st_precheck'.
  --
  -- Both were formed inline in 'st_geom_mul'/'st_div_w' as TWO chained
  -- 32-bit adds, then compared against the kernel size and subtracted from
  -- again, all in the cycle that also decides 'state'. Post-route
  -- 'desc_q[flags][3] -> state[*]' was 11-14 logic levels at -0.771 ns.
  -- Everything here fits in 18 bits ('in_width' is 16, each pad field 8),
  -- so this is also the "bound the arithmetic before registering it" half
  -- of 'shared/TimingAndResources.md' section 2, not just an extra stage.
  signal padded_w_q : unsigned(17 downto 0) := (others => '0');
  signal padded_h_q : unsigned(17 downto 0) := (others => '0');

  -- 'reserved_w0 /= 0 or reserved_w10 /= 0'.
  signal chk_reserved_bad_q : std_ulogic := '0';
  -- 'xfer_bytes /= 0'.
  signal chk_xfer_nz_q : std_ulogic := '0';
  -- Per-operand 8-byte alignment.
  signal chk_al_in_q : std_ulogic := '0';
  signal chk_al_out_q : std_ulogic := '0';
  signal chk_al_wgt_q : std_ulogic := '0';
  signal chk_al_bias_q : std_ulogic := '0';
  signal chk_al_scale_q : std_ulogic := '0';
  signal chk_al_xfer_q : std_ulogic := '0';
  -- Geometry predicates, already reduced.
  signal chk_dims_nz_q : std_ulogic := '0';
  signal chk_conv_geom_q : std_ulogic := '0';
  signal chk_pool_geom_q : std_ulogic := '0';

  signal busy_q : std_ulogic := '0';
  signal err_code_q : err_code_t := c_err_none;
  signal err_pc_q : unsigned(31 downto 0) := (others => '0');
  signal pending_err_q : err_code_t := c_err_none;

  ------------------------------------------------------------------------
  -- Geometry, all in bytes/words, all derived from the descriptor. Widths
  -- are deliberately generous (32 bit) and every product is 'resize'd
  -- before assignment, because 'numeric_std."*"' returns the sum of the
  -- operand widths and silently truncating that is the classic bug here.
  ------------------------------------------------------------------------

  signal n_tiles_q : unsigned(15 downto 0) := (others => '0');   -- T = ceil(Cin/8)
  signal n_planes_out_q : unsigned(15 downto 0) := (others => '0');
  signal n_ot_q : unsigned(15 downto 0) := (others => '0');      -- ceil(Cout/pe_rows)
  signal in_plane_bytes_q : unsigned(31 downto 0) := (others => '0');
  signal in_row_bytes_q : unsigned(31 downto 0) := (others => '0');
  signal in_total_bytes_q : unsigned(31 downto 0) := (others => '0');
  signal out_plane_bytes_q : unsigned(31 downto 0) := (others => '0');
  signal out_total_bytes_q : unsigned(31 downto 0) := (others => '0');
  -- 'out_w * out_h' and 'in_w * in_h', registered between the two
  -- geometry-output states so that neither total is two chained
  -- multiplies deep. 32 bits: 16x16 is exactly 32 in numeric_std.
  -- Source-extent verdict, carried from 'st_range' to 'st_range_dst'.
  signal range_err_q : err_code_t := c_err_none;

  -- Reserved-field/space-tag verdict, carried from 'st_validate' to
  -- 'st_validate2' -- same split as 'range_err_q' above and for the same
  -- reason (section 2): one 7-deep sequential 'v_err' chain per command
  -- is free logic-level budget spent for nothing when a cycle boundary
  -- is available, and this state has one.
  signal validate_err_q : err_code_t := c_err_none;

  signal out_plane_words_q : unsigned(31 downto 0) := (others => '0');
  signal in_plane_words_q : unsigned(31 downto 0) := (others => '0');

  signal out_w_q : unsigned(15 downto 0) := (others => '0');
  signal out_h_q : unsigned(15 downto 0) := (others => '0');
  signal row_words_q : unsigned(15 downto 0) := (others => '0');  -- in_width * T

  -- 'n_tiles_q' and 'row_words_q' less one, registered in the same cycle
  -- as the values themselves.
  --
  -- The tile-buffer control asks "last tile?" and "last word of the row?"
  -- once per beat, in four places between them, and each inline '- 1'
  -- rebuilds a 16-bit subtract out of a register that has not changed
  -- since 'st_geom_mul' decoded the descriptor. Routed P&R made the
  -- result the design's worst setup path: 'n_tiles_q' to the write
  -- address of 'rowbuf', nine logic levels of which five were CARRY4.
  --
  -- Both sources are written at exactly one place -- the 'st_geom_mul'
  -- state below -- so a copy written in that same cycle is not a
  -- pipeline stage and cannot skew against its source: the pair updates
  -- atomically, one clock edge, and every reader sees them together.
  -- (That is why only these two are hoisted and 'desc_q.in_width - 1'
  -- and friends are left alone: 'desc_q' is written elsewhere too, so a
  -- copy of it would need its own argument about when it is coherent.)
  --
  -- Wrap-around is preserved rather than avoided: for a zero source
  -- these hold x"FFFF", exactly what the inline '- 1' produced.
  --
  -- See shared/TimingAndResources.md 2, "Bound the arithmetic before
  -- registering it" -- the loop-invariant-hoisting half of it.
  signal n_tiles_m1_q : unsigned(15 downto 0) := (others => '0');
  signal row_words_m1_q : unsigned(15 downto 0) := (others => '0');
  signal wgt_tile_bytes_q : unsigned(31 downto 0) := (others => '0');
  -- 'kernel_h * kernel_w' and 'n_tiles * (kernel_h * kernel_w)', each on
  -- its own cycle.
  --
  -- 'wgt_tile_bytes' was 'n_tiles_q * kernel_h * kernel_w * (pe_rows *
  -- pe_cols)' -- three chained runtime products in one cycle, which Vivado
  -- mapped to a DSP cascade whose inter-DSP hop
  -- ('wgt_tile_bytes_q2 -> wgt_tile_bytes_q1', ZERO logic levels, -0.588 ns
  -- post-route) no placement could close. This is the same defect
  -- 'st_geom_out'/'st_geom_out2' above were already split for -- one
  -- product per cycle - applied to the one product chain that pass missed.
  -- ("When a fix works, search the whole design for the same pattern",
  -- 'shared/TimingAndResources.md'.) Values are unchanged: every stage is
  -- exact at its own width, and integer multiplication does not care about
  -- association order.
  signal kernel_area_q : unsigned(15 downto 0) := (others => '0');
  signal wgt_tile_taps_q : unsigned(31 downto 0) := (others => '0');

  -- Sequential restoring divider, shared by the two output-dimension
  -- divides (out_w then out_h). A runtime divide by 'stride' is
  -- unavoidable -- the ISA lets stride be any value -- and 16 shift/
  -- subtract steps once per command is far cheaper than a synthesised
  -- divider, so it is spelled out here rather than written as '/'.
  signal div_num_q : unsigned(31 downto 0) := (others => '0');
  signal div_den_q : unsigned(31 downto 0) := (others => '0');
  signal div_quot_q : unsigned(15 downto 0) := (others => '0');
  signal div_rem_q : unsigned(31 downto 0) := (others => '0');
  signal div_step_q : natural range 0 to 16 := 0;
  signal div_busy_q : std_ulogic := '0';
  -- Set for one handshake by the divide step when the quotient is complete,
  -- cleared by the FSM state that consumes it. See the note above 'main'.
  signal div_valid_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Loop counters: the OT loop (section 5.4) for conv, the plane loop for
  -- pooling. Both index the same "one pass over the whole ifmap" concept.
  ------------------------------------------------------------------------

  signal pass_q : unsigned(15 downto 0) := (others => '0');

  ------------------------------------------------------------------------
  -- Per-pass DDR byte offsets, kept as ACCUMULATORS rather than as
  -- 'pass_q * <stride>' products.
  --
  -- Each of the four DDR base addresses this FSM issues (weight tile,
  -- bias row, scale row, output plane, input plane) used to be formed as
  -- 'base + resize(pass_q * stride(15 downto 0), 32)' in the state that
  -- issues the request. 'pass_q' advances by exactly one per pass and
  -- every stride is a per-command constant, so the product is an
  -- accumulator by construction -- and building it as a product cost a
  -- whole DSP48 plus a 32-bit carry chain in the FSM's own critical
  -- cycle. Post-route that was the worst path in the accelerator:
  --
  --   'cmd_proc/wgt_tile_bytes_q1/CLK (DSP48E1)
  --    -> cmd_proc/side_req_q_reg[req][addr][31]/D'
  --   -1.040 ns, 10 logic levels (DSP48E1 + 8 CARRY4), 4.87 ns of logic
  --
  -- As accumulators the same addresses are one 32-bit add off a register,
  -- and the DSP and its clock-to-out disappear. Values are identical by
  -- induction: each offset starts at 0 exactly where 'pass_q' is cleared
  -- and gains its stride exactly where 'pass_q' is incremented, and the
  -- strides are the same '(15 downto 0)' truncations the products used,
  -- so every partial sum equals the product it replaces.
  --
  -- This is shared/TimingAndResources.md section 2 ("per-command
  -- configuration must never be computed per cycle") applied to address
  -- generation, and the same technique cnn_accel_window_gen already uses
  -- for its row-bank read address ('rd_base_q').
  ------------------------------------------------------------------------
  signal wgt_pass_off_q : unsigned(31 downto 0) := (others => '0');
  signal bias_pass_off_q : unsigned(31 downto 0) := (others => '0');
  signal scale_pass_off_q : unsigned(31 downto 0) := (others => '0');
  signal out_pass_off_q : unsigned(31 downto 0) := (others => '0');
  signal in_pass_off_q : unsigned(31 downto 0) := (others => '0');
  signal pass_last_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Space resolution for the in-flight command (registered at validate
  -- time so that every mux below is driven from state, never from a
  -- same-cycle decode of a signal another process is also writing).
  ------------------------------------------------------------------------

  signal src0_is_ddr_q : std_ulogic := '0';
  signal dst_is_ddr_q : std_ulogic := '0';
  signal dst_is_weight_q : std_ulogic := '0';
  signal side_is_ddr_q : std_ulogic := '0';   -- src1 / weights / LUT
  signal side_is_local_q : std_ulogic := '0'; -- ... in LOCAL_TENSOR
  signal engine_conv_q : std_ulogic := '0';
  signal engine_pool_q : std_ulogic := '0';
  signal engine_elem_q : std_ulogic := '0';
  signal xfer_active_q : std_ulogic := '0';
  -- 'LOAD'-style destination: section 4 dedicates write channel 'w0' to
  -- the DDR->LOCAL landing path and 'w1' to engine output, so a pure move
  -- into the scratchpad and a compute result never share a port.
  signal dst_use_w0_q : std_ulogic := '0';
  -- The side operand is the ACT LUT rather than weights/src1.
  signal side_is_lut_q : std_ulogic := '0';
  signal side_is_src1_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Internal, space-independent operand ports. The mux block at the end
  -- of the architecture binds each of these to one physical port
  -- according to the registers above.
  ------------------------------------------------------------------------

  signal src_req_q : dma_req_m2s_t := c_dma_req_init;
  -- The ifmap feeder's own source-request register. Kept separate from
  -- 'src_req_q' (which the main FSM owns for the LOAD/STORE/LOADW class)
  -- because the two live in different processes and a signal may have only
  -- one driver; they are never both in use, since the engine classes and the
  -- move class are disjoint.
  signal feed_req_q : dma_req_m2s_t := c_dma_req_init;
  signal src_req_ready : std_ulogic;
  signal src_done : std_ulogic;
  signal src_error : std_ulogic;
  signal src_m2s : axi_stream_m2s_t;
  signal src_ready : std_ulogic := '0';

  signal side_req_q : dma_req_m2s_t := c_dma_req_init;
  signal side_req_ready : std_ulogic;
  signal side_done : std_ulogic;
  signal side_error : std_ulogic;
  signal side_m2s : axi_stream_m2s_t;
  signal side_ready : std_ulogic := '0';

  signal dst_req_q : dma_req_m2s_t := c_dma_req_init;
  signal dst_req_ready : std_ulogic;
  signal dst_done : std_ulogic;
  signal dst_error : std_ulogic;
  signal dst_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal dst_ready : std_ulogic;

  ------------------------------------------------------------------------
  -- Ifmap feeder: issues the source read requests of one pass. For
  -- 'T = 1' that is a single whole-tensor request; for 'T > 1' it is one
  -- '(row, tile)' strip request at a time, feeding the transpose buffer
  -- (see the entity-level comment).
  ------------------------------------------------------------------------

  type feed_state_t is (fd_idle, fd_issue, fd_wait, fd_done);
  signal feed_state : feed_state_t := fd_idle;
  signal feed_kick : std_ulogic := '0';
  signal feed_row_q : unsigned(15 downto 0) := (others => '0');
  signal feed_tile_q : unsigned(15 downto 0) := (others => '0');

  -- The '(row, tile)' strip address, maintained as an ACCUMULATOR instead
  -- of being formed as 'in_addr + tile*in_plane_bytes + row*in_row_bytes'
  -- in the cycle the request is presented.
  --
  -- 'shared/TimingAndResources.md' section 2, "Never form a runtime product
  -- of configuration values in a datapath": both products change only at a
  -- tile/row boundary, so they are maintained by addition at that boundary.
  -- The old form was a DSP48E1 followed by two 32-bit adds
  -- ('feed_req_q[req][addr]' at -1.066 ns post-route, 11 levels of which 8
  -- were CARRY4); this form is a register read with nothing in front of it,
  -- and the two adds that remain each sit alone on a 'fd_wait' cycle that
  -- only ever happens once per strip.
  --
  -- Bit-exactness: the old expression truncated both stride operands to
  -- their low 16 bits before multiplying and took the whole sum mod 2**32,
  -- so the steps accumulated here are deliberately the same truncated
  -- values -- 'c_tile_step_i'/'c_row_step_i' below.
  signal feed_addr_q : unsigned(31 downto 0) := (others => '0');
  -- 'in_addr + row*in_row_bytes': the strip address at tile 0 of the
  -- current row, so a row advance does not have to undo the tile term.
  signal feed_row_base_q : unsigned(31 downto 0) := (others => '0');
  signal feed_tile_step_i : unsigned(31 downto 0);
  signal feed_row_step_i : unsigned(31 downto 0);
  signal feed_split_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Single-row transpose buffer. Depth is the same 'g_max_row_tile_words'
  -- bound 'cnn_accel_window_gen' already places on 'in_width * T', so a
  -- geometry that fits the window generator always fits here too and the
  -- 'c_err_bad_geometry' check that guards one guards both.
  ------------------------------------------------------------------------

  subtype word_t is std_ulogic_vector(8 * c_word_bytes - 1 downto 0);
  type rowbuf_t is array (0 to g_max_row_tile_words - 1) of word_t;
  signal rowbuf : rowbuf_t;

  signal tb_fill_q : std_ulogic := '0';
  signal tb_drain_q : std_ulogic := '0';
  signal tb_wr_ptr_q : unsigned(15 downto 0) := (others => '0');
  signal tb_rd_ptr_q : unsigned(15 downto 0) := (others => '0');
  signal tb_col_q : unsigned(15 downto 0) := (others => '0');
  signal tb_tile_q : unsigned(15 downto 0) := (others => '0');
  signal tb_row_q : unsigned(15 downto 0) := (others => '0');
  signal tb_last_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Weight-fill serializer. 'cnn_accel_weight_buffer's fill port takes
  -- exactly one lane per beat -- one int8 weight, one int32 bias, or one
  -- packed scale entry -- while the DDR/scratchpad image it is filled
  -- from is a dense byte image delivered 8 bytes per beat. The lane width
  -- is therefore runtime-selected (8 / 32 / 64 bits), which no fixed
  -- 'common.width_conversion' instance can do, so the unpacking is a
  -- small shift register here rather than three instances plus a mux.
  ------------------------------------------------------------------------

  type wgt_region_t is (rg_weight, rg_bias, rg_scale);
  signal wgt_region_q : wgt_region_t := rg_weight;
  -- The region tag that belongs to the word currently in 'wgt_hold_q'.
  -- 'wgt_region_q' is the region being *requested*, and it is advanced as
  -- soon as the next sub-region's request is armed -- which can happen
  -- while the serializer is still draining the previous region's last
  -- word. Tagging the outgoing beats from 'wgt_region_q' would therefore
  -- mislabel those trailing lanes (and shift the holding register by the
  -- wrong lane width), so the tag travels with the data instead.
  signal wgt_hold_region_q : wgt_region_t := rg_weight;
  signal wgt_hold_q : word_t := (others => '0');
  signal wgt_lanes_left_q : natural range 0 to 8 := 0;
  signal wgt_fill_active_q : std_ulogic := '0';
  signal wgt_stage_q : natural range 0 to 2 := 0;

  ------------------------------------------------------------------------
  -- Watchdog and counters.
  ------------------------------------------------------------------------

  -- Effective request ports: either this module's own registers or, while
  -- 'cnn_accel_elementwise' owns the command, that engine's request ports.
  signal src_req_eff : dma_req_m2s_t;
  signal side_req_eff : dma_req_m2s_t;
  signal dst_req_eff : dma_req_m2s_t;

  -- Engine-input stream after de-interleaving (conv) or straight through.
  signal eng_in_m2s : axi_stream_m2s_t;
  signal eng_in_ready : std_ulogic;
  signal tb_bypass : std_ulogic;

  -- Weight-fill serializer output.
  signal wgt_lane_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal wgt_lane_ready : std_ulogic;

  signal watchdog_q : unsigned(31 downto 0) := (others => '0');
  signal progress : std_ulogic;
  -- 'progress' REGISTERED, for the watchdog reload only.
  --
  -- 'progress' is an OR of ten handshake/done signals gathered from every
  -- engine and DMA in the accelerator, and the watchdog reload it drives
  -- lands on the R/CE pins of all 32 'watchdog_q' bits. Post-route that
  -- made 'pool_window_gen/n_res_q -> cmd_proc/watchdog_q[*]/R' the design's
  -- worst path (-1.419 ns, 11 levels, 71 % route) -- the cross-module
  -- ready-chain signature in 'shared/TimingAndResources.md' section 2:
  -- "a combinational 'ready' ... derived from a descriptor or status
  -- register several levels up and fanned out to hundreds of consumers".
  --
  -- Registering it costs the watchdog one cycle of reload latency out of
  -- 'g_watchdog_cycles' (2 000 in the fastest test configuration,
  -- 1 000 000 by default) and changes no observable behaviour: a stalled
  -- engine still produces no progress at all, and a running one still
  -- reloads far more often than the counter can expire. The performance
  -- counters below deliberately keep reading the combinational 'progress'
  -- so 'CNT_STALL' stays cycle-exact.
  signal progress_q : std_ulogic := '0';

  signal cnt_cmd_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_cycle_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_compute_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_stall_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_ddr_rd_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_ddr_wr_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_load_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_store_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_wgt_bytes_q : unsigned(31 downto 0) := (others => '0');
  signal cnt_local_rd_q : unsigned(41 downto 0) := (others => '0');
  signal cnt_local_wr_q : unsigned(41 downto 0) := (others => '0');

  ------------------------------------------------------------------------
  -- Helper functions. All pure decode/arithmetic on descriptor fields.
  ------------------------------------------------------------------------

  -- 'boolean' -> 'std_ulogic', so a predicate can be reduced to one
  -- registered bit in 'st_precheck'.
  function to_sl(value : boolean) return std_ulogic is
  begin
    if value then
      return '1';
    else
      return '0';
    end if;
  end function;

  function classify(opcode : std_ulogic_vector(7 downto 0)) return cmd_class_t is
  begin
    if opcode = c_opcode_halt then
      return cls_halt;
    elsif opcode = c_opcode_conv2d or opcode = c_opcode_fc then
      return cls_conv;
    elsif opcode = c_opcode_pool_max or opcode = c_opcode_pool_avg then
      return cls_pool;
    elsif opcode = c_opcode_load or opcode = c_opcode_store
      or opcode = c_opcode_loadw then
      return cls_xfer;
    elsif opcode = c_opcode_add or opcode = c_opcode_upsample
      or opcode = c_opcode_copy or opcode = c_opcode_act then
      return cls_elem;
    else
      -- Includes DWCONV2D, which is allocated but rejected (section 5.2).
      return cls_bad;
    end if;
  end function;

  -- True for the v1.2 opcodes, whose 'xfer_bytes' (W15) must be zero
  -- because in v1.2 that word was "reserved, must be 0" (section 5.1).
  function is_v12_opcode(opcode : std_ulogic_vector(7 downto 0)) return boolean is
  begin
    return opcode = c_opcode_halt or opcode = c_opcode_conv2d
      or opcode = c_opcode_dwconv2d or opcode = c_opcode_pool_max
      or opcode = c_opcode_pool_avg or opcode = c_opcode_fc;
  end function;

  function is_aligned(addr : unsigned) return boolean is
  begin
    return addr(c_word_shift - 1 downto 0) = 0;
  end function;

  -- ceil(value / 2**shift) for a power-of-two divisor.
  function ceil_shift(value : unsigned; shift : natural) return unsigned is
    constant c_round : unsigned(value'range) := to_unsigned(2 ** shift - 1, value'length);
  begin
    return shift_right(value + c_round, shift);
  end function;

begin

  ------------------------------------------------------------------------
  -- Elaboration-time contracts. Both are properties this module's byte
  -- arithmetic depends on and neither can be discovered at runtime.
  ------------------------------------------------------------------------

  -- One 'cnn_accel_bias_requant' output beat carries 'g_pe_rows' int8
  -- lanes and is written straight into the PLANES layout as one word, so
  -- an OT pass must be exactly one output plane. Lifting this needs a
  -- width conversion on the ofmap path, not a change here.
  assert g_pe_rows = c_plane_channels
    report "cnn_accel_cmd_proc: g_pe_rows must equal the activation plane " &
      "channel count (" & integer'image(c_plane_channels) & ") -- one OT pass " &
      "is one output plane"
    severity failure;

  assert g_tile_channels = c_plane_channels
    report "cnn_accel_cmd_proc: g_tile_channels must equal the activation " &
      "plane channel count -- one scratchpad/DDR word is one input-channel tile"
    severity failure;

  ------------------------------------------------------------------------
  -- Watchdog. 'progress' is every externally observable forward step; the
  -- absence of all of them for 'g_watchdog_cycles' cycles is what
  -- 'c_err_timeout' means (section 9).
  ------------------------------------------------------------------------

  progress <= (src_m2s.valid and src_ready)
    or (dst_m2s.valid and dst_ready)
    or (side_m2s.valid and side_ready)
    or src_done or dst_done or side_done
    or conv_done or pool_done or ew_done or fetch_desc_valid;

  ------------------------------------------------------------------------
  -- Main command FSM, with the sequential divider inlined at the top of it.
  --
  -- The divider ('div_quot_q = div_num_q / div_den_q', 16-step restoring) is
  -- deliberately not a process of its own: the FSM loads the operands, clears
  -- the accumulators and consumes the quotient, so FSM and divider write the
  -- same registers and two processes would mean two drivers. Putting the
  -- divide step first in the same process gives the FSM's assignments
  -- last-writer priority on the cycles where both fire, which is exactly the
  -- wanted "load/clear beats shift" behaviour.
  --
  -- 'div_valid_q' is what makes the handshake safe for a zero result. An
  -- "are all the registers still clear?" start condition would restart the
  -- divide forever whenever quotient and remainder both came out zero -- a
  -- legal case, 'in_width = kernel_w' gives a zero dividend -- and since
  -- 'st_div_w'/'st_div_h' pet the watchdog on every cycle, that livelock
  -- would never be broken.
  ------------------------------------------------------------------------

  main : process(clk)
    variable v_err : err_code_t;
    variable v_cls : cmd_class_t;
    variable v_len : unsigned(31 downto 0);
    variable v_prod : unsigned(31 downto 0);
    variable v_ok : boolean;
    variable v_rem : unsigned(31 downto 0);
  begin
    if rising_edge(clk) then
      ------------------------------------------------------------------
      -- One restoring-division step.
      ------------------------------------------------------------------
      if div_busy_q = '1' then
        v_rem := div_rem_q(30 downto 0) & div_num_q(15);
        div_num_q <= div_num_q(30 downto 0) & '0';
        if v_rem >= div_den_q then
          div_rem_q <= v_rem - div_den_q;
          div_quot_q <= div_quot_q(14 downto 0) & '1';
        else
          div_rem_q <= v_rem;
          div_quot_q <= div_quot_q(14 downto 0) & '0';
        end if;

        if div_step_q = 15 then
          div_busy_q <= '0';
          div_step_q <= 0;
          div_valid_q <= '1';
        else
          div_step_q <= div_step_q + 1;
        end if;
      end if;

      -- Default: every pulse output is one cycle wide.
      fetch_start <= '0';
      fetch_desc_ready <= '0';
      conv_start <= '0';
      pool_start <= '0';
      ew_start <= '0';
      seq_done <= '0';
      seq_error <= '0';
      feed_kick <= '0';

      progress_q <= progress;

      if watchdog_q /= 0 then
        watchdog_q <= watchdog_q - 1;
      end if;
      if progress_q = '1' then
        watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);
      end if;

      case state is

        --------------------------------------------------------------
        when st_idle =>
          busy_q <= '0';
          if start = '1' then
            pc_q <= unsigned(program_base_addr);
            busy_q <= '1';
            state <= st_fetch;
            watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);
          end if;

        --------------------------------------------------------------
        -- Descriptor fetch. 'cnn_accel_cmd_fetch' owns the burst; this
        -- state only hands it a PC and waits for the decoded record.
        when st_fetch =>
          fetch_start <= '1';
          fetch_addr <= pc_q;
          state <= st_fetch_wait;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        when st_fetch_wait =>
          if fetch_error = '1' then
            pending_err_q <= fetch_error_code;
            state <= st_error;
          elsif fetch_desc_valid = '1' then
            fetch_desc_ready <= '1';
            desc_q <= fetch_desc;
            pc_q <= fetch_pc;
            cls_q <= classify(fetch_desc.opcode);
            state <= st_precheck;
            watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);
          end if;

        --------------------------------------------------------------
        -- Pre-decode. Every wide comparison 'st_validate' needs, reduced
        -- to one registered bit, and the opcode decoded to one-hot flags.
        -- Reads only 'desc_q' (registered last cycle) and writes only
        -- registers; no state transition logic depends on any of it.
        when st_precheck =>
          op_is_add_q <= to_sl(desc_q.opcode = c_opcode_add);
          op_is_act_q <= to_sl(desc_q.opcode = c_opcode_act);
          op_is_loadw_q <= to_sl(desc_q.opcode = c_opcode_loadw);
          op_is_copy_q <= to_sl(desc_q.opcode = c_opcode_copy);
          op_is_v12_q <= to_sl(is_v12_opcode(desc_q.opcode));
          op_is_upsample_q <= to_sl(desc_q.opcode = c_opcode_upsample);

          padded_w_q <= resize(desc_q.in_width, 18);
          padded_h_q <= resize(desc_q.in_height, 18);
          if desc_q.flags(c_flag_pad_en) = '1' then
            padded_w_q <= resize(desc_q.in_width, 18)
              + desc_q.pad_left + desc_q.pad_right;
            padded_h_q <= resize(desc_q.in_height, 18)
              + desc_q.pad_top + desc_q.pad_bottom;
          end if;

          chk_reserved_bad_q <=
            to_sl(desc_q.reserved_w0 /= x"00" or desc_q.reserved_w10 /= x"0000");
          chk_xfer_nz_q <= to_sl(desc_q.xfer_bytes /= 0);

          chk_al_in_q <= to_sl(is_aligned(desc_q.in_addr));
          chk_al_out_q <= to_sl(is_aligned(desc_q.out_addr));
          chk_al_wgt_q <= to_sl(is_aligned(desc_q.weight_addr));
          chk_al_bias_q <= to_sl(is_aligned(desc_q.bias_addr));
          chk_al_scale_q <= to_sl(is_aligned(desc_q.scale_addr));
          chk_al_xfer_q <= to_sl(is_aligned(desc_q.xfer_bytes));

          chk_dims_nz_q <= to_sl(
            desc_q.in_width /= 0 and desc_q.in_height /= 0
            and desc_q.in_channels /= 0
          );
          chk_conv_geom_q <= to_sl(
            desc_q.out_channels /= 0
            and desc_q.kernel_h /= 0 and desc_q.kernel_w /= 0
            and desc_q.stride_h /= 0 and desc_q.stride_w /= 0
            and desc_q.kernel_h <= g_max_kernel_size
            and desc_q.kernel_w <= g_max_kernel_size
          );
          chk_pool_geom_q <= to_sl(
            desc_q.pool_kernel_h /= 0 and desc_q.pool_kernel_w /= 0
            and desc_q.pool_stride_h /= 0 and desc_q.pool_stride_w /= 0
            and desc_q.pool_kernel_h <= g_max_pool_kernel_size
            and desc_q.pool_kernel_w <= g_max_pool_kernel_size
          );

          state <= st_validate;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Validation, part 1: everything decidable from the descriptor
        -- alone (section 9). Part 2 (address ranges) needs the byte
        -- counts and runs in 'st_range'.
        when st_validate =>
          v_err := c_err_none;
          v_cls := cls_q;

          -- Opcode legality.
          if v_cls = cls_bad then
            v_err := c_err_unsupported_op;
          end if;

          -- Reserved fields (section 5.1): both 'must be 0' gaps, plus
          -- 'xfer_bytes', which was itself a reserved word in v1.2 and so
          -- must still be zero on a v1.2 opcode. Rejecting a non-zero
          -- reserved field is what keeps the compatibility promise
          -- enforceable in both directions: a v2.0 program that sets bits
          -- this revision does not understand is refused loudly instead of
          -- being executed with those bits quietly ignored, which is what
          -- would turn a future ISA extension into a silent wrong answer
          -- on old hardware.
          if v_err = c_err_none
            and (chk_reserved_bad_q = '1'
              or (op_is_v12_q = '1' and chk_xfer_nz_q = '1')) then
            v_err := c_err_bad_reserved;
          end if;

          -- Space-tag legality (section 3). 'c_space_reserved' is always
          -- illegal. 'LOCAL_WEIGHT' is legal only on the weight/bias/
          -- scale operand group, plus on 'LOADW's destination, which is
          -- by definition a write into that group.
          if v_err = c_err_none then
            if desc_q.space_src0 = c_space_reserved
              or desc_q.space_dst = c_space_reserved
              or desc_q.space_wgt = c_space_reserved
              or (v_cls = cls_elem and op_is_add_q = '1'
                  and desc_q.space_src1 = c_space_reserved) then
              v_err := c_err_bad_space;
            elsif desc_q.space_src0 = c_space_local_weight then
              v_err := c_err_bad_space;
            elsif desc_q.space_dst = c_space_local_weight
              and op_is_loadw_q = '0' then
              v_err := c_err_bad_space;
            elsif op_is_add_q = '1'
              and desc_q.space_src1 = c_space_local_weight then
              v_err := c_err_bad_space;
            end if;
          end if;

          -- Carry the verdict so far into 'st_validate2' -- see
          -- 'validate_err_q's declaration comment -- rather than chaining
          -- the alignment/geometry/xfer-bytes checks and the dispatch off
          -- this same 'v_err' in the same cycle.
          validate_err_q <= v_err;
          state <= st_validate2;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Validation, part 1b: alignment, geometry, operand-space
        -- resolution and the class dispatch -- split from 'st_validate'
        -- above (same reasoning as 'st_range'/'st_range_dst' below: one
        -- deep chain of checks off 'cls_q' and 'desc_q', now two shallower
        -- ones. Per command, so the extra cycle is free.
        when st_validate2 =>
          v_err := validate_err_q;
          v_cls := cls_q;

          -- Alignment (section 3). Every operand address the command will
          -- actually use, plus its byte count, must be a whole number of
          -- 8-byte words -- both spaces are word-granular and neither
          -- DMA can express a sub-word burst.
          if v_err = c_err_none and v_cls /= cls_halt then
            v_ok := chk_al_in_q = '1' and chk_al_out_q = '1';
            if v_cls = cls_conv then
              v_ok := v_ok and chk_al_wgt_q = '1'
                and chk_al_bias_q = '1'
                and chk_al_scale_q = '1';
            end if;
            if op_is_add_q = '1' then
              v_ok := v_ok and chk_al_xfer_q = '1';
            elsif op_is_act_q = '1' then
              v_ok := v_ok and chk_al_wgt_q = '1'
                and chk_al_xfer_q = '1';
            elsif op_is_v12_q = '0' then
              v_ok := v_ok and chk_al_xfer_q = '1';
            end if;
            if not v_ok then
              v_err := c_err_misaligned;
            end if;
          end if;

          -- Geometry sanity (section 9). Only the dimensions each class
          -- actually consumes are checked, so a 'LOAD' is not rejected
          -- for leaving 'in_width' at zero.
          if v_err = c_err_none and (v_cls = cls_conv or v_cls = cls_pool) then
            v_ok := chk_dims_nz_q = '1';
            if v_cls = cls_conv then
              v_ok := v_ok and chk_conv_geom_q = '1';
            else
              v_ok := v_ok and chk_pool_geom_q = '1';
            end if;
            if not v_ok then
              v_err := c_err_bad_geometry;
            end if;
          end if;

          if v_err = c_err_none and v_cls = cls_elem
            and op_is_copy_q = '0' and op_is_act_q = '0' then
            if chk_dims_nz_q = '0' then
              v_err := c_err_bad_geometry;
            end if;
          end if;

          if v_err = c_err_none and v_cls = cls_xfer and chk_xfer_nz_q = '0' then
            v_err := c_err_bad_geometry;
          end if;

          -- Resolve the operand spaces once, here, so that every mux is
          -- driven from a register for the whole life of the command.
          src0_is_ddr_q <= '0';
          if desc_q.space_src0 = c_space_ddr then
            src0_is_ddr_q <= '1';
          end if;
          dst_is_ddr_q <= '0';
          dst_is_weight_q <= '0';
          if desc_q.space_dst = c_space_ddr then
            dst_is_ddr_q <= '1';
          elsif desc_q.space_dst = c_space_local_weight then
            dst_is_weight_q <= '1';
          end if;
          -- The side operand is 'src1' for ADD and the weight/bias/scale/
          -- LUT group for everything else; they never coexist.
          side_is_ddr_q <= '0';
          side_is_local_q <= '0';
          side_is_src1_q <= '0';
          side_is_lut_q <= '0';
          if op_is_add_q = '1' then
            side_is_src1_q <= '1';
          elsif op_is_act_q = '1' then
            side_is_lut_q <= '1';
          end if;
          if op_is_add_q = '1' then
            if desc_q.space_src1 = c_space_ddr then
              side_is_ddr_q <= '1';
            elsif desc_q.space_src1 = c_space_local_tensor then
              side_is_local_q <= '1';
            end if;
          else
            if desc_q.space_wgt = c_space_ddr then
              side_is_ddr_q <= '1';
            elsif desc_q.space_wgt = c_space_local_tensor then
              side_is_local_q <= '1';
            end if;
          end if;

          engine_conv_q <= '0';
          engine_pool_q <= '0';
          engine_elem_q <= '0';
          xfer_active_q <= '0';
          dst_use_w0_q <= '0';
          if v_cls = cls_xfer and desc_q.space_dst = c_space_local_tensor then
            dst_use_w0_q <= '1';
          end if;
          case v_cls is
            when cls_conv => engine_conv_q <= '1';
            when cls_pool => engine_pool_q <= '1';
            when cls_elem => engine_elem_q <= '1';
            when cls_xfer => xfer_active_q <= '1';
            when others => null;
          end case;

          if v_err /= c_err_none then
            pending_err_q <= v_err;
            state <= st_error;
          elsif v_cls = cls_halt then
            state <= st_retire;
          else
            state <= st_geom_mul;
          end if;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Geometry, step 1: the products that do not need a divide.
        when st_geom_mul =>
          n_tiles_q <= resize(
            ceil_shift(desc_q.in_channels, c_word_shift), n_tiles_q'length
          );
          -- See the declaration: the tile-buffer's "last tile?" tests read
          -- this instead of rebuilding the subtract every beat.
          n_tiles_m1_q <= resize(
            ceil_shift(desc_q.in_channels, c_word_shift), n_tiles_q'length
          ) - 1;
          n_ot_q <= resize(
            ceil_shift(desc_q.out_channels, c_word_shift), n_ot_q'length
          );

          -- in_plane_bytes = in_width * in_height * 8. The product is
          -- resized explicitly: 16x16 multiplication is 32 bits wide in
          -- numeric_std and the shift must not push bits off the top.
          v_prod := resize(desc_q.in_width * desc_q.in_height, v_prod'length);
          in_plane_bytes_q <= shift_left(v_prod, c_word_shift);
          in_row_bytes_q <= shift_left(resize(desc_q.in_width, 32), c_word_shift);
          row_words_q <= resize(
            desc_q.in_width * ceil_shift(desc_q.in_channels, c_word_shift),
            row_words_q'length
          );
          row_words_m1_q <= resize(
            desc_q.in_width * ceil_shift(desc_q.in_channels, c_word_shift),
            row_words_q'length
          ) - 1;
          -- First of the three 'wgt_tile_bytes' products.
          kernel_area_q <= resize(desc_q.kernel_h * desc_q.kernel_w, 16);

          -- Set up the first divide: out_w = (in_w + pad_l + pad_r - k_w)
          -- / stride_w + 1, or the pooling equivalent. 'pad_en' gates the
          -- padding fields exactly as the golden model does.
          if cls_q = cls_conv then
            if padded_w_q < desc_q.kernel_w then
              pending_err_q <= c_err_bad_geometry;
              state <= st_error;
            else
              div_num_q <= resize(padded_w_q - desc_q.kernel_w, 32);
              div_den_q <= resize(desc_q.stride_w, 32);
              state <= st_div_w;
            end if;
          else
            -- ISA v2.1: pooling is padded too, with the same fields and
            -- the same 'pad_en' gate 'padded_w_q' already applied.
            if padded_w_q < desc_q.pool_kernel_w then
              pending_err_q <= c_err_bad_geometry;
              state <= st_error;
            else
              div_num_q <= resize(padded_w_q - desc_q.pool_kernel_w, 32);
              div_den_q <= resize(desc_q.pool_stride_w, 32);
              state <= st_div_w;
            end if;
          end if;

          if cls_q = cls_xfer then
            -- A pure move is described entirely by 'xfer_bytes': no
            -- shape, no divide, and nothing in 'st_geom_out'/'out2' that
            -- its validation reads. It still goes through 'st_range_sum':
            -- that is where BOTH extents are formed, and skipping it would
            -- range-check this command against the previous command's
            -- addresses.
            state <= st_range_sum;
          elsif cls_q = cls_elem then
            -- No output-dimension divide either -- an elementwise
            -- command's output is the same shape as its input -- but it
            -- must NOT skip 'st_geom_out'/'st_geom_out2': those are
            -- where 'in_plane_words' and 'in_total_bytes' are computed,
            -- and 'st_range'/'st_range_dst' validate ADD and UPSAMPLE
            -- against 'in_total_bytes'. Jumping straight to 'st_range'
            -- left that register holding the PREVIOUS command's ifmap
            -- size, so an ADD following a larger convolution was
            -- range-checked against a length that had nothing to do with
            -- it -- and raised 'ERR_LOCAL_RANGE' on a perfectly legal
            -- descriptor. Invisible in the untiled catalogue, where every
            -- tensor in a program is the same shape; the first tiled case
            -- with a per-plane ADD after a taller conv strip hit it
            -- immediately.
            --
            -- 'out_w_q'/'out_h_q' are stale on this path and the products
            -- 'st_geom_out'/'out2' derive from them ('out_plane_bytes',
            -- 'out_total_bytes', 'n_planes_out') are meaningless -- and
            -- unread: the elementwise engine sequences its own transfers
            -- and never enters the pass loop, and the 'out_w = 0'
            -- geometry check is gated on conv/pool. Only the two input
            -- products matter here, and they are computed from the
            -- descriptor's own fields.
            state <= st_geom_out;
          end if;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        when st_div_w =>
          if div_busy_q = '0' and div_valid_q = '0' then
            div_busy_q <= '1';
            div_step_q <= 0;
            div_quot_q <= (others => '0');
            div_rem_q <= (others => '0');
          elsif div_valid_q = '1' then
            out_w_q <= div_quot_q + 1;
            div_valid_q <= '0';
            div_quot_q <= (others => '0');
            div_rem_q <= (others => '0');
            if cls_q = cls_conv then
              if padded_h_q < desc_q.kernel_h then
                pending_err_q <= c_err_bad_geometry;
                state <= st_error;
              else
                div_num_q <= resize(padded_h_q - desc_q.kernel_h, 32);
                div_den_q <= resize(desc_q.stride_h, 32);
                state <= st_div_h;
              end if;
            else
              if padded_h_q < desc_q.pool_kernel_h then
                pending_err_q <= c_err_bad_geometry;
                state <= st_error;
              else
                div_num_q <= resize(padded_h_q - desc_q.pool_kernel_h, 32);
                div_den_q <= resize(desc_q.pool_stride_h, 32);
                state <= st_div_h;
              end if;
            end if;
          end if;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        when st_div_h =>
          if div_busy_q = '0' and div_valid_q = '0' then
            div_busy_q <= '1';
            div_step_q <= 0;
            div_quot_q <= (others => '0');
            div_rem_q <= (others => '0');
          elsif div_valid_q = '1' then
            out_h_q <= div_quot_q + 1;
            div_valid_q <= '0';
            div_quot_q <= (others => '0');
            div_rem_q <= (others => '0');
            state <= st_geom_out;
          end if;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Geometry, step 2a: the output plane itself. Nothing that needs a
        -- SECOND multiply on top of this product happens here -- chaining
        -- 'out_w * out_h' into '* n_ot' (and 'in_w * in_h' into
        -- '* n_tiles') put two runtime multipliers in one cycle and cost
        -- 3.1 ns of setup slack at 150 MHz. Both second products moved to
        -- 'st_geom_out2', off the register written here. One more cycle
        -- per command.
        when st_geom_out =>
          v_prod := resize(out_w_q * out_h_q, v_prod'length);
          out_plane_words_q <= v_prod;
          out_plane_bytes_q <= shift_left(v_prod, c_word_shift);
          in_plane_words_q <= resize(desc_q.in_width * desc_q.in_height, v_prod'length);

          state <= st_geom_out2;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Geometry, step 2b: the totals, each one product off a register.
        when st_geom_out2 =>
          v_prod := out_plane_words_q;

          if cls_q = cls_conv then
            -- One OT pass writes exactly one output plane (asserted
            -- above), so the whole ofmap is 'n_ot' planes.
            out_total_bytes_q <= shift_left(
              resize(v_prod * n_ot_q, 32), c_word_shift
            );
            n_planes_out_q <= n_ot_q;
            -- One weight tile is, per 'pack_weights_for_hw'
            -- (T x k_h x k_w x pe_rows x pe_cols int8), this many bytes.
            -- Second product; the third is in 'st_range_sum'.
            wgt_tile_taps_q <= resize(n_tiles_q * kernel_area_q, 32);
            pass_q <= (others => '0');
            -- Pass-offset accumulators start with 'pass_q' -- see their
            -- declaration.
            wgt_pass_off_q <= (others => '0');
            bias_pass_off_q <= (others => '0');
            scale_pass_off_q <= (others => '0');
            out_pass_off_q <= (others => '0');
            in_pass_off_q <= (others => '0');
          else
            out_total_bytes_q <= shift_left(
              resize(v_prod * n_tiles_q, 32), c_word_shift
            );
            n_planes_out_q <= n_tiles_q;
            pass_q <= (others => '0');
            -- Pass-offset accumulators start with 'pass_q' -- see their
            -- declaration.
            wgt_pass_off_q <= (others => '0');
            bias_pass_off_q <= (others => '0');
            scale_pass_off_q <= (others => '0');
            out_pass_off_q <= (others => '0');
            in_pass_off_q <= (others => '0');
          end if;

          in_total_bytes_q <= shift_left(
            resize(in_plane_words_q * n_tiles_q, 32),
            c_word_shift
          );

          state <= st_range_sum;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Range check, step 0: the two extents. Nothing but the operand
        -- length select and one 33-bit add each; the comparisons and the
        -- error-priority chain that consume them are in the two states
        -- below. All of it is per-command.
        when st_range_sum =>
          -- Source extent.
          if cls_q = cls_xfer or cls_q = cls_elem then
            v_len := desc_q.xfer_bytes;
            if cls_q = cls_elem
              and (op_is_add_q = '1' or op_is_upsample_q = '1') then
              v_len := in_total_bytes_q;
            end if;
          else
            v_len := in_total_bytes_q;
          end if;
          src_end_q <= resize(desc_q.in_addr, 33) + v_len;

          -- Destination extent.
          if cls_q = cls_xfer then
            v_len := desc_q.xfer_bytes;
          elsif cls_q = cls_elem then
            v_len := desc_q.xfer_bytes;
            if op_is_add_q = '1' then
              v_len := in_total_bytes_q;
            elsif op_is_upsample_q = '1' then
              -- Nearest-2x2 quadruples the pixel count (section 5.2).
              v_len := shift_left(in_total_bytes_q, 2);
            end if;
          else
            v_len := out_total_bytes_q;
          end if;
          dst_end_q <= resize(desc_q.out_addr, 33) + v_len;

          -- Third and last 'wgt_tile_bytes' product. Landing it here is
          -- still three states ahead of 'st_wgt_setup', the first reader.
          wgt_tile_bytes_q <= resize(
            wgt_tile_taps_q * to_unsigned(g_pe_rows * g_pe_cols, 16),
            wgt_tile_bytes_q'length
          );

          state <= st_range;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Validation, part 2: address ranges, now that the byte counts
        -- exist. 'LOCAL_TENSOR' is bounded by 'g_tensor_bytes', 'DDR' by
        -- 'g_ddr_limit' (sections 3 and 6).
        when st_range =>
          -- Source extent only. The destination extent, the geometry
          -- bounds and the class dispatch move to 'st_range_dst': doing
          -- all of them in one cycle was a 19-level chain of 33-bit adds
          -- and comparisons off 'cls_q' and 'desc_q', and one of the
          -- accelerator's worst remaining paths at 150 MHz. Per command,
          -- so the extra cycle is free.
          v_err := c_err_none;

          if desc_q.space_src0 = c_space_local_tensor then
            if src_end_q > g_tensor_bytes then
              v_err := c_err_local_range;
            end if;
          elsif desc_q.space_src0 = c_space_ddr then
            if src_end_q > g_ddr_limit then
              v_err := c_err_ddr_range;
            end if;
          end if;

          range_err_q <= v_err;
          state <= st_range_dst;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- Validation, part 2b: destination extent, the row-bank and
        -- output-dimension bounds, and the dispatch.
        when st_range_dst =>
          v_err := range_err_q;

          if v_err = c_err_none then
            if desc_q.space_dst = c_space_local_tensor then
              if dst_end_q > g_tensor_bytes then
                v_err := c_err_local_range;
              end if;
            elsif desc_q.space_dst = c_space_ddr then
              if dst_end_q > g_ddr_limit then
                v_err := c_err_ddr_range;
              end if;
            end if;
          end if;

          -- 'cnn_accel_window_gen's own row-bank bound, and the transpose
          -- buffer's, are the same number (section 4 / entity comment).
          if v_err = c_err_none and cls_q = cls_conv
            and row_words_q > g_max_row_tile_words then
            v_err := c_err_bad_geometry;
          end if;
          if v_err = c_err_none and (cls_q = cls_conv or cls_q = cls_pool)
            and (out_w_q = 0 or out_h_q = 0) then
            v_err := c_err_bad_geometry;
          end if;

          if v_err /= c_err_none then
            pending_err_q <= v_err;
            state <= st_error;
          else
            case cls_q is
              when cls_conv => state <= st_wgt_setup;
              when cls_pool => state <= st_pass_setup;
              when cls_xfer => state <= st_xfer_req_src;
              when cls_elem =>
                -- One-cycle dispatch pulse, landing on the first cycle of
                -- 'st_elem_run'; the scalar command fields are already
                -- stable (they are a pure function of 'desc_q').
                ew_start <= '1';
                state <= st_elem_run;
              when others => state <= st_retire;
            end case;
          end if;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        -- CONV2D/FC weight refill for output-channel tile 'pass_q'
        -- (section 5.4). Skipped entirely when 'FLAG_WEIGHT_REUSE' is set
        -- (invariant R6: zero added to 'WEIGHT_LOAD_BYTES') or when the
        -- weights already live in 'LOCAL_WEIGHT'.
        when st_wgt_setup =>
          if desc_q.flags(c_flag_weight_reuse) = '1'
            or desc_q.space_wgt = c_space_local_weight then
            state <= st_pass_setup;
          else
            wgt_stage_q <= 0;
            wgt_region_q <= rg_weight;
            conv_fill_start <= '1';
            state <= st_wgt_req;
          end if;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        when st_wgt_req =>
          conv_fill_start <= '0';
          -- One request per sub-region, in the order the weight buffer
          -- expects (weights, then bias, then scale): each region's
          -- pointer is independent, so an empty region is simply skipped.
          if side_req_q.valid = '0' then
            case wgt_stage_q is
              when 0 =>
                wgt_region_q <= rg_weight;
                side_req_q.req.addr <= desc_q.weight_addr + wgt_pass_off_q;
                side_req_q.req.length <= wgt_tile_bytes_q;
                side_req_q.valid <= '1';
                wgt_fill_active_q <= '1';
                cnt_wgt_bytes_q <= cnt_wgt_bytes_q + wgt_tile_bytes_q;
              when 1 =>
                if desc_q.flags(c_flag_bias_en) = '1' then
                  wgt_region_q <= rg_bias;
                  side_req_q.req.addr <= desc_q.bias_addr + bias_pass_off_q;
                  side_req_q.req.length <=
                    to_unsigned(g_pe_rows * c_bias_entry_bytes, 32);
                  side_req_q.valid <= '1';
                  wgt_fill_active_q <= '1';
                  cnt_wgt_bytes_q <= cnt_wgt_bytes_q
                    + to_unsigned(g_pe_rows * c_bias_entry_bytes, 32);
                else
                  wgt_stage_q <= 2;
                end if;
              when others =>
                if desc_q.flags(c_flag_per_channel_en) = '1' then
                  wgt_region_q <= rg_scale;
                  side_req_q.req.addr <= desc_q.scale_addr + scale_pass_off_q;
                  side_req_q.req.length <=
                    to_unsigned(g_pe_rows * c_scale_entry_bytes, 32);
                  side_req_q.valid <= '1';
                  wgt_fill_active_q <= '1';
                  cnt_wgt_bytes_q <= cnt_wgt_bytes_q
                    + to_unsigned(g_pe_rows * c_scale_entry_bytes, 32);
                else
                  state <= st_pass_setup;
                end if;
            end case;
          elsif side_req_ready = '1' then
            side_req_q.valid <= '0';
            state <= st_wgt_run;
          end if;

        when st_wgt_run =>
          if side_error = '1' then
            pending_err_q <= c_err_axi;
            state <= st_error;
          elsif side_done = '1' then
            wgt_fill_active_q <= '0';
            if wgt_stage_q = 2 then
              state <= st_pass_setup;
            else
              wgt_stage_q <= wgt_stage_q + 1;
              state <= st_wgt_req;
            end if;
          end if;

        --------------------------------------------------------------
        -- One pass over the ifmap: an OT tile for conv, one activation
        -- plane for pooling. The destination write job is issued first so
        -- that no engine output beat can ever arrive before its sink is
        -- armed, then the engine is started and the feeder kicked.
        when st_pass_setup =>
          if pass_q = n_planes_out_q - 1 then
            pass_last_q <= '1';
          else
            pass_last_q <= '0';
          end if;

          dst_req_q.req.addr <= desc_q.out_addr + out_pass_off_q;
          dst_req_q.req.length <= out_plane_bytes_q;
          dst_req_q.valid <= '1';
          state <= st_pass_req_dst;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        when st_pass_req_dst =>
          if dst_req_ready = '1' then
            dst_req_q.valid <= '0';
            if engine_conv_q = '1' then
              conv_start <= '1';
            else
              pool_start <= '1';
            end if;
            feed_kick <= '1';
            state <= st_pass_run;
          end if;

        when st_pass_run =>
          if src_error = '1' or dst_error = '1' then
            pending_err_q <= c_err_axi;
            state <= st_error;
          elsif (engine_conv_q = '1' and conv_done = '1')
            or (engine_pool_q = '1' and pool_done = '1') then
            state <= st_pass_drain;
          end if;

        when st_pass_drain =>
          -- The engine has emitted its last beat; the sink still has to
          -- retire it. Waiting for 'dst_done' rather than the engine's
          -- own 'done' is what makes 'DDR_WR_BYTES' exact.
          if dst_error = '1' then
            pending_err_q <= c_err_axi;
            state <= st_error;
          elsif dst_done = '1' then
            if pass_last_q = '1' then
              state <= st_retire;
            else
              pass_q <= pass_q + 1;
              -- ... and advance with it, by one stride each. See their
              -- declaration for why this is exactly the product it
              -- replaces.
              wgt_pass_off_q <= wgt_pass_off_q
                + resize(wgt_tile_bytes_q(15 downto 0), 32);
              bias_pass_off_q <= bias_pass_off_q
                + to_unsigned(g_pe_rows * c_bias_entry_bytes, 32);
              scale_pass_off_q <= scale_pass_off_q
                + to_unsigned(g_pe_rows * c_scale_entry_bytes, 32);
              out_pass_off_q <= out_pass_off_q
                + resize(out_plane_bytes_q(15 downto 0), 32);
              in_pass_off_q <= in_pass_off_q
                + resize(in_plane_bytes_q(15 downto 0), 32);
              if engine_conv_q = '1' then
                state <= st_wgt_setup;
              else
                state <= st_pass_setup;
              end if;
            end if;
          end if;

        --------------------------------------------------------------
        -- LOAD / STORE / LOADW: a single linear move of 'xfer_bytes'
        -- from the source space to the destination space. The two request
        -- ports are armed back to back and the stream is then a straight
        -- pass-through, so the direction is genuinely just the pair of
        -- space tags (section 5.2).
        when st_xfer_req_src =>
          if src_req_q.valid = '0' then
            src_req_q.req.addr <= desc_q.in_addr;
            src_req_q.req.length <= desc_q.xfer_bytes;
            src_req_q.valid <= '1';
          elsif src_req_ready = '1' then
            src_req_q.valid <= '0';
            state <= st_xfer_req_dst;
          end if;

        when st_xfer_req_dst =>
          if dst_is_weight_q = '1' then
            -- LOADW: the sink is the weight buffer's fill port, which has
            -- no request handshake -- only a fill-session pulse, and only
            -- for the weight region (the bias/scale pointers must not be
            -- rewound by a bias-only or scale-only LOADW; see
            -- cnn_accel_weight_buffer's own fill_start contract).
            wgt_fill_active_q <= '1';
            if desc_q.flags(c_flag_per_channel_en) = '1' then
              wgt_region_q <= rg_scale;
            elsif desc_q.flags(c_flag_bias_en) = '1' then
              wgt_region_q <= rg_bias;
            else
              wgt_region_q <= rg_weight;
              conv_fill_start <= '1';
            end if;
            cnt_wgt_bytes_q <= cnt_wgt_bytes_q + desc_q.xfer_bytes;
            state <= st_xfer_run;
          elsif dst_req_q.valid = '0' then
            dst_req_q.req.addr <= desc_q.out_addr;
            dst_req_q.req.length <= desc_q.xfer_bytes;
            dst_req_q.valid <= '1';
          elsif dst_req_ready = '1' then
            dst_req_q.valid <= '0';
            state <= st_xfer_run;
          end if;

        when st_xfer_run =>
          conv_fill_start <= '0';
          if src_error = '1' or dst_error = '1' then
            pending_err_q <= c_err_axi;
            state <= st_error;
          elsif dst_is_weight_q = '1' then
            if src_done = '1' then
              wgt_fill_active_q <= '0';
              state <= st_retire;
            end if;
          elsif dst_done = '1' then
            state <= st_retire;
          end if;

        --------------------------------------------------------------
        -- ADD / UPSAMPLE / COPY / ACT. 'cnn_accel_elementwise' owns its
        -- own request/stream sequencing; this module only supplies the
        -- scalar command fields and binds its four ports to the physical
        -- ones the space tags select.
        when st_elem_run =>
          if ew_error = '1' then
            pending_err_q <= ew_error_code;
            state <= st_error;
          elsif ew_done = '1' then
            state <= st_retire;
          end if;

        --------------------------------------------------------------
        when st_retire =>
          cnt_cmd_q <= cnt_cmd_q + 1;
          if desc_q.opcode = c_opcode_load then
            cnt_load_q <= cnt_load_q + 1;
          elsif desc_q.opcode = c_opcode_store then
            cnt_store_q <= cnt_store_q + 1;
          end if;

          if cls_q = cls_halt then
            state <= st_done;
          else
            pc_q <= desc_q.next_instr_addr;
            -- A descriptor pointing at itself, or at a misaligned/out-of-
            -- range address, must not be able to spin forever: the former
            -- is caught by the watchdog, the latter here.
            if not is_aligned(desc_q.next_instr_addr) then
              pending_err_q <= c_err_misaligned;
              state <= st_error;
            elsif resize(desc_q.next_instr_addr, 33) + c_instr_word_bytes
              > g_ddr_limit then
              pending_err_q <= c_err_ddr_range;
              state <= st_error;
            else
              state <= st_fetch;
            end if;
          end if;
          watchdog_q <= to_unsigned(g_watchdog_cycles, watchdog_q'length);

        --------------------------------------------------------------
        when st_done =>
          seq_done <= '1';
          busy_q <= '0';
          state <= st_idle;

        --------------------------------------------------------------
        when st_error =>
          seq_error <= '1';
          err_code_q <= pending_err_q;
          err_pc_q <= pc_q;
          busy_q <= '0';
          -- Every operand port is released; nothing of a rejected command
          -- is ever left half-issued (section 9).
          src_req_q.valid <= '0';
          dst_req_q.valid <= '0';
          side_req_q.valid <= '0';
          wgt_fill_active_q <= '0';
          state <= st_idle;

      end case;

      --------------------------------------------------------------
      -- The watchdog is the single termination guarantee for every wait
      -- above (section 9). 'st_idle'/'st_done'/'st_error' are excluded
      -- because they do not wait on anything.
      if watchdog_q = 0 and busy_q = '1'
        and state /= st_idle and state /= st_done and state /= st_error then
        pending_err_q <= c_err_timeout;
        state <= st_error;
      end if;

      if reset = '1' or soft_reset_pulse = '1' then
        state <= st_idle;
        busy_q <= '0';
        div_busy_q <= '0';
        div_step_q <= 0;
        div_valid_q <= '0';
        src_req_q.valid <= '0';
        dst_req_q.valid <= '0';
        side_req_q.valid <= '0';
        wgt_fill_active_q <= '0';
        fetch_start <= '0';
        fetch_desc_ready <= '0';
        conv_start <= '0';
        pool_start <= '0';
        ew_start <= '0';
        conv_fill_start <= '0';
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Ifmap feeder. One request at a time, waiting for each to retire
  -- before issuing the next, so at most one source burst is ever in
  -- flight and the 'src_done' pulse is unambiguous.
  ------------------------------------------------------------------------

  feed_tile_step_i <= resize(in_plane_bytes_q(15 downto 0), 32);
  feed_row_step_i <= resize(in_row_bytes_q(15 downto 0), 32);

  feeder : process(clk)
  begin
    if rising_edge(clk) then
      case feed_state is
        when fd_idle =>
          if feed_kick = '1' then
            feed_row_q <= (others => '0');
            feed_tile_q <= (others => '0');
            feed_addr_q <= desc_q.in_addr;
            feed_row_base_q <= desc_q.in_addr;
            if engine_conv_q = '1' and n_tiles_q > 1 then
              feed_split_q <= '1';
            else
              feed_split_q <= '0';
            end if;
            feed_state <= fd_issue;
          end if;

        when fd_issue =>
          if feed_req_q.valid = '0' then
            if feed_split_q = '1' then
              -- One '(row, tile)' strip: plane 'tile' is
              -- 'in_plane_bytes' away, row 'row' is 'in_row_bytes' into it.
              feed_req_q.req.addr <= feed_addr_q;
              feed_req_q.req.length <= in_row_bytes_q;
            elsif engine_pool_q = '1' then
              feed_req_q.req.addr <= desc_q.in_addr + in_pass_off_q;
              feed_req_q.req.length <= in_plane_bytes_q;
            else
              feed_req_q.req.addr <= desc_q.in_addr;
              feed_req_q.req.length <= in_total_bytes_q;
            end if;
            feed_req_q.valid <= '1';
          elsif src_req_ready = '1' then
            feed_req_q.valid <= '0';
            feed_state <= fd_wait;
          end if;

        when fd_wait =>
          if src_done = '1' then
            if feed_split_q = '0' then
              feed_state <= fd_done;
            elsif feed_tile_q = n_tiles_m1_q then
              feed_tile_q <= (others => '0');
              if feed_row_q = desc_q.in_height - 1 then
                feed_state <= fd_done;
              else
                feed_row_q <= feed_row_q + 1;
                -- Row advance: back to tile 0 of the next row. One add,
                -- shared between both registers.
                feed_row_base_q <= feed_row_base_q + feed_row_step_i;
                feed_addr_q <= feed_row_base_q + feed_row_step_i;
                feed_state <= fd_issue;
              end if;
            else
              feed_tile_q <= feed_tile_q + 1;
              feed_addr_q <= feed_addr_q + feed_tile_step_i;
              feed_state <= fd_issue;
            end if;
          end if;

        when fd_done =>
          feed_state <= fd_idle;
      end case;

      if reset = '1' or soft_reset_pulse = '1' or state = st_error then
        feed_state <= fd_idle;
        feed_req_q.valid <= '0';
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Ifmap transpose buffer (see the entity-level comment). Only ever
  -- engaged for a convolution with 'T > 1'; everything else takes the
  -- bypass, which is a plain wire.
  ------------------------------------------------------------------------

  tb_bypass <= '1' when engine_conv_q = '0' or n_tiles_q <= 1 else '0';

  transpose : process(clk)
  begin
    if rising_edge(clk) then
      if tb_bypass = '0' then
        -- Fill: the feeder delivers one '(row, tile)' strip at a time, so
        -- consecutive accepted beats land at consecutive cells and cell
        -- 'tile * in_width + col' ends up holding exactly that tile's
        -- column, which is what the drain side indexes.
        if tb_fill_q = '1' and src_m2s.valid = '1' then
          rowbuf(to_integer(tb_wr_ptr_q)) <= src_m2s.data(word_t'range);
          if tb_wr_ptr_q = row_words_m1_q then
            tb_wr_ptr_q <= (others => '0');
            tb_fill_q <= '0';
            tb_drain_q <= '1';
            tb_col_q <= (others => '0');
            tb_tile_q <= (others => '0');
            tb_rd_ptr_q <= (others => '0');
          else
            tb_wr_ptr_q <= tb_wr_ptr_q + 1;
          end if;
        end if;

        -- Drain in '[x][t]' order, which is what 'cnn_accel_window_gen'
        -- ingests. The read pointer walks by 'in_width' within a pixel and
        -- restarts at the next column when the pixel's tiles are done, so
        -- no multiplier is needed on the critical path.
        if tb_drain_q = '1' and m_conv_stream_s2m.ready = '1' then
          if tb_tile_q = n_tiles_m1_q then
            tb_tile_q <= (others => '0');
            if tb_col_q = desc_q.in_width - 1 then
              -- Row complete: back to filling, unless the frame is done.
              tb_drain_q <= '0';
              tb_col_q <= (others => '0');
              tb_rd_ptr_q <= (others => '0');
              if tb_row_q = desc_q.in_height - 1 then
                tb_row_q <= (others => '0');
              else
                tb_row_q <= tb_row_q + 1;
                tb_fill_q <= '1';
              end if;
            else
              tb_col_q <= tb_col_q + 1;
              tb_rd_ptr_q <= resize(tb_col_q + 1, tb_rd_ptr_q'length);
            end if;
          else
            tb_tile_q <= tb_tile_q + 1;
            tb_rd_ptr_q <= tb_rd_ptr_q + desc_q.in_width;
          end if;
        end if;
      end if;

      -- A new pass always restarts the buffer, so a pass aborted by an
      -- error or a soft reset cannot leave half a row behind.
      if feed_kick = '1' then
        tb_wr_ptr_q <= (others => '0');
        tb_rd_ptr_q <= (others => '0');
        tb_col_q <= (others => '0');
        tb_tile_q <= (others => '0');
        tb_row_q <= (others => '0');
        tb_drain_q <= '0';
        tb_fill_q <= '1';
      end if;

      if reset = '1' or soft_reset_pulse = '1' or state = st_error then
        tb_fill_q <= '0';
        tb_drain_q <= '0';
      end if;
    end if;
  end process;

  -- Final beat of the frame: last tile of the last column of the last row.
  tb_last_q <= '1' when tb_drain_q = '1' and tb_tile_q = n_tiles_m1_q
    and tb_col_q = desc_q.in_width - 1 and tb_row_q = desc_q.in_height - 1
    else '0';

  ------------------------------------------------------------------------
  -- Engine input stream: the transpose buffer's drain side for a tiled
  -- convolution, the raw source stream otherwise.
  ------------------------------------------------------------------------

  eng_in_mux : process(all)
  begin
    if tb_bypass = '1' then
      eng_in_m2s <= src_m2s;
    else
      eng_in_m2s <= axi_stream_m2s_init;
      eng_in_m2s.valid <= tb_drain_q;
      eng_in_m2s.data <= (others => '0');
      eng_in_m2s.data(word_t'range) <= rowbuf(to_integer(tb_rd_ptr_q));
      eng_in_m2s.last <= tb_last_q;
      eng_in_m2s.user <= (others => '0');
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Weight-fill serializer (see its declaration comment). Runtime lane
  -- width: one int8 weight, one int32 bias, or one packed scale entry per
  -- beat out, from one 8-byte word in.
  ------------------------------------------------------------------------

  weight_serializer : process(clk)
  begin
    if rising_edge(clk) then
      if wgt_lanes_left_q = 0 then
        if wgt_fill_active_q = '1' and side_m2s.valid = '1' then
          wgt_hold_q <= side_m2s.data(word_t'range);
          wgt_hold_region_q <= wgt_region_q;
          case wgt_region_q is
            when rg_weight => wgt_lanes_left_q <= c_word_bytes;
            when rg_bias => wgt_lanes_left_q <= c_word_bytes / c_bias_entry_bytes;
            when rg_scale => wgt_lanes_left_q <= c_word_bytes / c_scale_entry_bytes;
          end case;
        end if;
      elsif m_conv_weight_s2m.ready = '1' then
        wgt_lanes_left_q <= wgt_lanes_left_q - 1;
        case wgt_hold_region_q is
          when rg_weight =>
            wgt_hold_q <= std_ulogic_vector(shift_right(unsigned(wgt_hold_q), 8));
          when rg_bias =>
            wgt_hold_q <= std_ulogic_vector(
              shift_right(unsigned(wgt_hold_q), 8 * c_bias_entry_bytes)
            );
          when rg_scale => wgt_hold_q <= (others => '0');
        end case;
      end if;

      if reset = '1' or soft_reset_pulse = '1' or state = st_error then
        wgt_lanes_left_q <= 0;
      end if;
    end if;
  end process;

  wgt_lane_m2s.valid <= '1' when wgt_lanes_left_q /= 0 else '0';
  wgt_lane_m2s.last <= '0';
  wgt_lane_m2s.user <= (others => '0');

  -- Region tag presented to 'cnn_accel_weight_buffer' alongside every fill
  -- beat: it samples these in the same cycle as the beat (see that entity's
  -- 'fill_is_bias'/'fill_is_scale' contract), so they are decoded from the
  -- tag that travels with the held word, not from the request-side region.
  conv_fill_is_bias <= '1' when wgt_hold_region_q = rg_bias else '0';
  conv_fill_is_scale <= '1' when wgt_hold_region_q = rg_scale else '0';
  -- The lane always sits in the low bits of the holding register, so the
  -- region only decides how far the register is shifted after a beat, not
  -- where the beat is read from -- 'cnn_accel_weight_buffer' itself picks
  -- the width it cares about out of 'data'.
  wgt_lane_data : process(all)
  begin
    wgt_lane_m2s.data <= (others => '0');
    wgt_lane_m2s.data(word_t'range) <= wgt_hold_q;
  end process;

  ------------------------------------------------------------------------
  -- Operand-port binding (the table in the entity-level comment). Each
  -- internal operand is bound to exactly one physical port for the whole
  -- life of a command, so there is no arbitration here -- only selection.
  ------------------------------------------------------------------------

  -- Requests: this module's own for everything except an elementwise
  -- command, where the engine sequences its own transfers.
  -- Who owns the source port: the elementwise engine addresses DDR/scratchpad
  -- itself, the ifmap feeder drives the conv/pool classes, and the main FSM
  -- drives the LOAD/STORE/LOADW move class. Exactly one of the three can be
  -- active for a given command.
  src_req_eff <=
    ew_src0_req_m2s when engine_elem_q = '1'
    else feed_req_q when (engine_conv_q or engine_pool_q) = '1'
    else src_req_q;
  dst_req_eff <= ew_dst_req_m2s when engine_elem_q = '1' else dst_req_q;

  side_req_sel : process(all)
  begin
    if engine_elem_q = '1' and side_is_src1_q = '1' then
      side_req_eff <= ew_src1_req_m2s;
    elsif engine_elem_q = '1' and side_is_lut_q = '1' then
      side_req_eff <= ew_lut_req_m2s;
    else
      side_req_eff <= side_req_q;
    end if;
  end process;

  -- src0: DDR -> 'load' read DMA, LOCAL_TENSOR -> scratchpad read 0.
  load_req_m2s <= src_req_eff when src0_is_ddr_q = '1' else c_dma_req_init;
  tm_r0_req_m2s <= src_req_eff when src0_is_ddr_q = '0' else c_dma_req_init;
  src_req_ready <= load_req_s2m.ready when src0_is_ddr_q = '1' else tm_r0_req_s2m.ready;
  src_done <= load_dma_done when src0_is_ddr_q = '1' else tm_r0_done;
  src_error <= load_resp_error when src0_is_ddr_q = '1' else '0';
  src_m2s <= s_load_stream_m2s when src0_is_ddr_q = '1' else s_tm_r0_m2s;

  -- side operand (src1 / weights / LUT): DDR -> 'wgt' read DMA,
  -- LOCAL_TENSOR -> scratchpad read 1, LOCAL_WEIGHT -> already resident.
  wgt_req_m2s <= side_req_eff when side_is_ddr_q = '1' else c_dma_req_init;
  tm_r1_req_m2s <= side_req_eff when side_is_local_q = '1' else c_dma_req_init;
  side_req_ready <= wgt_req_s2m.ready when side_is_ddr_q = '1' else tm_r1_req_s2m.ready;
  side_done <= wgt_dma_done when side_is_ddr_q = '1' else tm_r1_done;
  side_error <= wgt_resp_error when side_is_ddr_q = '1' else '0';
  side_m2s <= s_wgt_stream_m2s when side_is_ddr_q = '1' else s_tm_r1_m2s;

  -- dst: DDR -> ofmap DMA, LOCAL_TENSOR -> scratchpad write 0 (pure move)
  -- or write 1 (engine output), LOCAL_WEIGHT -> the weight buffer's fill
  -- port, which has no request handshake at all.
  store_req_m2s <= dst_req_eff when dst_is_ddr_q = '1' else c_dma_req_init;
  tm_w0_req_m2s <= dst_req_eff when dst_use_w0_q = '1' else c_dma_req_init;
  tm_w1_req_m2s <= dst_req_eff
    when dst_is_ddr_q = '0' and dst_is_weight_q = '0' and dst_use_w0_q = '0'
    else c_dma_req_init;

  dst_port_sel : process(all)
  begin
    if dst_is_ddr_q = '1' then
      dst_req_ready <= store_req_s2m.ready;
      dst_done <= store_dma_done;
      dst_error <= store_resp_error;
      dst_ready <= m_store_stream_s2m.ready;
    elsif dst_use_w0_q = '1' then
      dst_req_ready <= tm_w0_req_s2m.ready;
      dst_done <= tm_w0_done;
      dst_error <= '0';
      dst_ready <= m_tm_w0_s2m.ready;
    elsif dst_is_weight_q = '1' then
      -- 'LOADW': the sink is the serializer, whose back-pressure is the
      -- weight buffer's own.
      dst_req_ready <= '1';
      dst_done <= '0';
      dst_error <= '0';
      dst_ready <= '1' when wgt_lanes_left_q = 0 else '0';
    else
      dst_req_ready <= tm_w1_req_s2m.ready;
      dst_done <= tm_w1_done;
      dst_error <= '0';
      dst_ready <= m_tm_w1_s2m.ready;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Data-stream binding. 'dst_m2s' is whatever the in-flight command's
  -- producer is; 'src_ready'/'side_ready' are whatever its consumer is.
  ------------------------------------------------------------------------

  producer_sel : process(all)
  begin
    if engine_elem_q = '1' then
      dst_m2s <= s_ew_dst_stream_m2s;
    elsif engine_conv_q = '1' then
      dst_m2s <= s_conv_out_m2s;
    elsif engine_pool_q = '1' then
      dst_m2s <= s_pool_out_m2s;
    else
      -- Pure move: the source stream is the destination stream.
      dst_m2s <= src_m2s;
    end if;
  end process;

  consumer_sel : process(all)
  begin
    if engine_elem_q = '1' then
      src_ready <= m_ew_src0_stream_s2m.ready;
    elsif engine_conv_q = '1' then
      if tb_bypass = '1' then
        src_ready <= m_conv_stream_s2m.ready;
      else
        src_ready <= tb_fill_q;
      end if;
    elsif engine_pool_q = '1' then
      src_ready <= m_pool_stream_s2m.ready;
    elsif xfer_active_q = '1' then
      src_ready <= dst_ready;
    else
      src_ready <= '0';
    end if;
  end process;

  side_ready_sel : process(all)
  begin
    if engine_elem_q = '1' and side_is_src1_q = '1' then
      side_ready <= m_ew_src1_stream_s2m.ready;
    elsif engine_elem_q = '1' and side_is_lut_q = '1' then
      side_ready <= m_ew_lut_stream_s2m.ready;
    elsif wgt_fill_active_q = '1' then
      side_ready <= '1' when wgt_lanes_left_q = 0 else '0';
    else
      side_ready <= '0';
    end if;
  end process;

  -- Back-pressure towards the physical read ports.
  s_load_stream_s2m.ready <= src_ready when src0_is_ddr_q = '1' else '0';
  s_tm_r0_s2m.ready <= src_ready when src0_is_ddr_q = '0' else '0';
  s_wgt_stream_s2m.ready <= side_ready when side_is_ddr_q = '1' else '0';
  s_tm_r1_s2m.ready <= side_ready when side_is_local_q = '1' else '0';

  -- Forward towards the physical write ports. Only the selected sink ever
  -- sees 'valid', which is what makes invariant R1 structural: a command
  -- with a LOCAL destination cannot produce an AXI write beat.
  store_fanout : process(all)
  begin
    m_store_stream_m2s <= dst_m2s;
    m_tm_w0_m2s <= dst_m2s;
    m_tm_w1_m2s <= dst_m2s;
    m_store_stream_m2s.valid <= dst_m2s.valid and dst_is_ddr_q;
    m_tm_w0_m2s.valid <= dst_m2s.valid and dst_use_w0_q;
    m_tm_w1_m2s.valid <= dst_m2s.valid and not (dst_is_ddr_q or dst_use_w0_q
      or dst_is_weight_q);
  end process;

  -- Engine input fan-out. Same rule: only the engine that owns the
  -- command ever sees a valid beat.
  eng_in_ready <= m_conv_stream_s2m.ready when engine_conv_q = '1'
    else m_pool_stream_s2m.ready;

  engine_fanout : process(all)
  begin
    m_conv_stream_m2s <= eng_in_m2s;
    m_pool_stream_m2s <= eng_in_m2s;
    m_ew_src0_stream_m2s <= src_m2s;
    m_conv_stream_m2s.valid <= eng_in_m2s.valid and engine_conv_q;
    m_pool_stream_m2s.valid <= eng_in_m2s.valid and engine_pool_q;
    m_ew_src0_stream_m2s.valid <= src_m2s.valid and engine_elem_q;
  end process;

  ew_side_fanout : process(all)
  begin
    m_ew_src1_stream_m2s <= side_m2s;
    m_ew_lut_stream_m2s <= side_m2s;
    m_ew_src1_stream_m2s.valid <= side_m2s.valid and engine_elem_q and side_is_src1_q;
    m_ew_lut_stream_m2s.valid <= side_m2s.valid and engine_elem_q and side_is_lut_q;
  end process;

  -- The weight buffer's fill port is fed either by the serializer (weight
  -- refill or 'LOADW') or by nothing at all.
  m_conv_weight_m2s <= wgt_lane_m2s;
  wgt_lane_ready <= m_conv_weight_s2m.ready;

  -- Engine output back-pressure.
  s_conv_out_s2m.ready <= dst_ready when engine_conv_q = '1' else '0';
  s_pool_out_s2m.ready <= dst_ready when engine_pool_q = '1' else '0';
  s_ew_dst_stream_s2m.ready <= dst_ready when engine_elem_q = '1' else '0';

  -- Request-handshake acknowledgements back to 'cnn_accel_elementwise'.
  ew_src0_req_s2m.ready <= src_req_ready when engine_elem_q = '1' else '0';
  ew_dst_req_s2m.ready <= dst_req_ready when engine_elem_q = '1' else '0';
  ew_src1_req_s2m.ready <= side_req_ready
    when engine_elem_q = '1' and side_is_src1_q = '1' else '0';
  ew_lut_req_s2m.ready <= side_req_ready
    when engine_elem_q = '1' and side_is_lut_q = '1' else '0';

  ------------------------------------------------------------------------
  -- Scalar command fields towards the engines. All pure functions of the
  -- latched descriptor, so they are stable for the whole command and no
  -- engine can sample a half-updated configuration.
  ------------------------------------------------------------------------

  -- REGISTERED, not combinational off 'desc_q'.
  --
  -- 'shared/ModernVHDL.md', "Per-command configuration boundary" /
  -- 'shared/TimingAndResources.md' section 2: "Put a register stage on
  -- every engine's configuration boundary. No engine's combinational cone
  -- may begin at the controller's descriptor register." Driven straight
  -- from 'desc_q' these nets ran from one register in 'cnn_accel_cmd_proc'
  -- to consumers placed wherever their own engine sits -- 'pool_cfg_opcode'
  -- into all eight 'pool_lane_gen' clock enables, 'pool_cfg_requant_scale'
  -- into 'pool_requant's DSP A/B ports. Post-route those were 100+ of the
  -- worst 400 endpoints, several of them at ZERO logic levels: pure route,
  -- with no placement that could satisfy them because launch and capture
  -- flip-flop were fixed at opposite ends of the path.
  --
  -- A register here gives the placer a flip-flop it can put next to each
  -- consumer group (and replicate). It costs one cycle of configuration
  -- latency, which is free by construction: every field below is a pure
  -- function of 'desc_q', latched in 'st_fetch_wait', while the earliest
  -- 'start' pulse any engine can see is issued in 'st_range_dst' -- seven
  -- states later. No engine can sample a stale value.
  cfg_out : process(clk)
  begin
    if rising_edge(clk) then
      geom_out_width <= std_ulogic_vector(out_w_q);
      geom_out_height <= std_ulogic_vector(out_h_q);

      conv_cfg_kernel_h <= std_ulogic_vector(desc_q.kernel_h);
      conv_cfg_kernel_w <= std_ulogic_vector(desc_q.kernel_w);
      conv_cfg_stride_h <= std_ulogic_vector(desc_q.stride_h);
      conv_cfg_stride_w <= std_ulogic_vector(desc_q.stride_w);
      conv_cfg_in_width <= std_ulogic_vector(desc_q.in_width);
      conv_cfg_in_height <= std_ulogic_vector(desc_q.in_height);
      conv_cfg_in_channels <= std_ulogic_vector(desc_q.in_channels);

      -- 'FLAG_PAD_EN' gates the four padding fields as one group, exactly as
      -- the golden model does; a descriptor with padding values but the flag
      -- clear convolves unpadded.
      conv_cfg_pad_top <= (others => '0');
      conv_cfg_pad_bottom <= (others => '0');
      conv_cfg_pad_left <= (others => '0');
      conv_cfg_pad_right <= (others => '0');
      if desc_q.flags(c_flag_pad_en) = '1' then
        conv_cfg_pad_top <= std_ulogic_vector(desc_q.pad_top);
        conv_cfg_pad_bottom <= std_ulogic_vector(desc_q.pad_bottom);
        conv_cfg_pad_left <= std_ulogic_vector(desc_q.pad_left);
        conv_cfg_pad_right <= std_ulogic_vector(desc_q.pad_right);
      end if;

      -- Ungated, see the port comment.
      conv_cfg_pad_value <= std_ulogic_vector(desc_q.pad_value);

      -- Fusion (section 5.3): the epilogue is configuration on the compute
      -- engine, never a second command, so the int32 accumulator tensor is
      -- never materialised anywhere this module can see.
      conv_cfg_bias_en <= desc_q.flags(c_flag_bias_en);
      conv_cfg_requant_en <= desc_q.flags(c_flag_requant_en);
      conv_cfg_relu_en <= desc_q.flags(c_flag_relu_en);
      conv_cfg_clamp_en <= desc_q.flags(c_flag_clamp_en);
      conv_cfg_per_channel_en <= desc_q.flags(c_flag_per_channel_en);
      conv_cfg_requant_scale <= std_ulogic_vector(desc_q.requant_scale);
      conv_cfg_requant_shift <= std_ulogic_vector(desc_q.requant_shift);
      conv_cfg_output_offset <= std_ulogic_vector(desc_q.output_offset);
      conv_cfg_clamp_min <= std_ulogic_vector(desc_q.clamp_min);
      conv_cfg_clamp_max <= std_ulogic_vector(desc_q.clamp_max);

      pool_cfg_kernel_h <= std_ulogic_vector(desc_q.pool_kernel_h);
      pool_cfg_kernel_w <= std_ulogic_vector(desc_q.pool_kernel_w);
      pool_cfg_stride_h <= std_ulogic_vector(desc_q.pool_stride_h);
      pool_cfg_stride_w <= std_ulogic_vector(desc_q.pool_stride_w);
      pool_cfg_pad_top <= x"00";
      if desc_q.flags(c_flag_pad_en) = '1' then
        pool_cfg_pad_top <= std_ulogic_vector(desc_q.pad_top);
      end if;
      pool_cfg_pad_bottom <= x"00";
      if desc_q.flags(c_flag_pad_en) = '1' then
        pool_cfg_pad_bottom <= std_ulogic_vector(desc_q.pad_bottom);
      end if;
      pool_cfg_pad_left <= x"00";
      if desc_q.flags(c_flag_pad_en) = '1' then
        pool_cfg_pad_left <= std_ulogic_vector(desc_q.pad_left);
      end if;
      pool_cfg_pad_right <= x"00";
      if desc_q.flags(c_flag_pad_en) = '1' then
        pool_cfg_pad_right <= std_ulogic_vector(desc_q.pad_right);
      end if;
      pool_cfg_pad_value <= std_ulogic_vector(desc_q.pad_value);
      pool_cfg_in_width <= std_ulogic_vector(desc_q.in_width);
      pool_cfg_in_height <= std_ulogic_vector(desc_q.in_height);
      pool_cfg_opcode <= desc_q.opcode;
      pool_cfg_requant_scale <= std_ulogic_vector(desc_q.requant_scale);
      pool_cfg_requant_shift <= std_ulogic_vector(desc_q.requant_shift);

      ew_opcode <= desc_q.opcode;
      ew_src0_addr <= desc_q.in_addr;
      -- W15 is a union: 'src1_addr' for ADD, a byte count for everything else
      -- (section 5.1, which is authoritative over section 5.2's prose).
      ew_src1_addr <= desc_q.xfer_bytes;
      ew_dst_addr <= desc_q.out_addr;
      ew_lut_addr <= desc_q.weight_addr;
      ew_xfer_bytes <= desc_q.xfer_bytes;
      ew_in_width <= desc_q.in_width;
      ew_in_height <= desc_q.in_height;
      ew_in_channels <= desc_q.in_channels;
      ew_requant_scale <= desc_q.requant_scale;
      ew_requant_shift <= desc_q.requant_shift;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Counters (section 8). Every one of them is an exact event count, not
  -- an estimate: 'DDR_*_BYTES' come from 'cnn_accel_axi_mux's per-cycle
  -- handshake measurement, 'LOCAL_*' from the scratchpad's own accepted
  -- beats, and the command counters from retirement.
  ------------------------------------------------------------------------

  counting : process(clk)
  begin
    if rising_edge(clk) then
      if busy_q = '1' then
        cnt_cycle_q <= cnt_cycle_q + 1;
        if state = st_pass_run or state = st_xfer_run or state = st_elem_run then
          cnt_compute_q <= cnt_compute_q + 1;
          if progress = '0' then
            cnt_stall_q <= cnt_stall_q + 1;
          end if;
        end if;
      end if;

      cnt_ddr_rd_q <= cnt_ddr_rd_q + axi_rd_bytes;
      cnt_ddr_wr_q <= cnt_ddr_wr_q + axi_wr_bytes;

      if (s_tm_r0_m2s.valid and s_tm_r0_s2m.ready) = '1' then
        cnt_local_rd_q <= cnt_local_rd_q + c_word_bytes;
      end if;
      if (s_tm_r1_m2s.valid and s_tm_r1_s2m.ready) = '1' then
        cnt_local_rd_q <= cnt_local_rd_q + c_word_bytes;
      end if;
      if (m_tm_w0_m2s.valid and m_tm_w0_s2m.ready) = '1' then
        cnt_local_wr_q <= cnt_local_wr_q + c_word_bytes;
      end if;
      if (m_tm_w1_m2s.valid and m_tm_w1_s2m.ready) = '1' then
        cnt_local_wr_q <= cnt_local_wr_q + c_word_bytes;
      end if;

      -- Cleared at START, not at reset, so a host can read the previous
      -- run's counters right up until it launches the next one.
      if start = '1' or reset = '1' then
        cnt_cycle_q <= (others => '0');
        cnt_compute_q <= (others => '0');
        cnt_stall_q <= (others => '0');
        cnt_ddr_rd_q <= (others => '0');
        cnt_ddr_wr_q <= (others => '0');
        cnt_local_rd_q <= (others => '0');
        cnt_local_wr_q <= (others => '0');
      end if;
    end if;
  end process;

  counters.cmd_count <= std_ulogic_vector(cnt_cmd_q);
  counters.cycle_count <= std_ulogic_vector(cnt_cycle_q);
  counters.compute_cycles <= std_ulogic_vector(cnt_compute_q);
  counters.stall_cycles <= std_ulogic_vector(cnt_stall_q);
  counters.ddr_rd_bytes <= std_ulogic_vector(cnt_ddr_rd_q);
  counters.ddr_wr_bytes <= std_ulogic_vector(cnt_ddr_wr_q);
  counters.tensor_load_count <= std_ulogic_vector(cnt_load_q);
  counters.tensor_store_count <= std_ulogic_vector(cnt_store_q);
  counters.weight_load_bytes <= std_ulogic_vector(cnt_wgt_bytes_q);
  counters.local_rd_kib <= std_ulogic_vector(cnt_local_rd_q(41 downto 10));
  counters.local_wr_kib <= std_ulogic_vector(cnt_local_wr_q(41 downto 10));

  err_code <= std_ulogic_vector(err_code_q);
  err_pc <= std_ulogic_vector(err_pc_q);

end architecture a;
