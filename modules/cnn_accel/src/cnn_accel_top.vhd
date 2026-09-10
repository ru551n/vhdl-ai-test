library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;
use cnn_accel.cnn_accel_v2_pkg.all;
use cnn_accel.cnn_accel_regs_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_lite;
use axi_lite.axi_lite_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library common;

-- Structural integration of the rev-2 programmable tensor accelerator, per
-- modules/cnn_accel/doc/cnn_accel_top_v2_arch.md section 2. This entity owns
-- no command, opcode or neural-network semantics of its own: everything that
-- decides *what* to do lives in 'cnn_accel_cmd_proc', everything that decides
-- *how* a tensor is computed lives in the engines. What is left here is
-- wiring, the two clock-domain-free adapters that the reused blocks need
-- (section "Pool engine" below), and the generic algebra that turns the
-- array/scratchpad geometry into the per-block generics.
--
-- Externally the accelerator is a single AXI4 master towards DDR plus a
-- single AXI4-Lite slave for the host:
--
--   host  --AXI4-Lite-->  cnn_accel_csr  --start/desc-->  cnn_accel_cmd_proc
--   cmd_proc  --engines/scratchpad-->  conv_core | pool | elementwise
--   cmd_proc  --DMA requests-->  3x axi_read_dma + 1x ofmap_dma
--   those 4 DMA masters  --cnn_accel_axi_mux-->  m_axi (DDR)
--
-- Reset: 'reset' is the cold, host-visible reset. 'reset_internal' additionally
-- folds in the CSR's 'soft_reset_pulse' and is what every datapath block sees,
-- so a soft reset aborts an in-flight command everywhere at once while leaving
-- the CSR (and therefore the error/status a host is about to read) intact. This
-- is the 'reset_internal' referred to by the reused blocks' port comments.
entity cnn_accel_top is
  generic (
    ----------------------------------------------------------------------
    -- Systolic-array geometry. 'g_pe_rows' output channels are produced in
    -- parallel (one output plane per OT pass, spec section 5.4) and
    -- 'g_pe_cols' input channels are consumed in parallel.
    ----------------------------------------------------------------------
    g_pe_rows : positive := cnn_accel_constant_pe_rows;
    g_pe_cols : positive := cnn_accel_constant_pe_cols;
    -- Input channels per activation-plane word. 'cnn_accel_conv_core' wants
    -- this equal to 'g_pe_cols', 'cnn_accel_cmd_proc' wants it equal to the
    -- generated activation-plane channel count; both are asserted below.
    g_tile_channels : positive := cnn_accel_constant_tile_channels;
    ----------------------------------------------------------------------
    -- Local tensor scratchpad geometry. 'g_tensor_bytes' -- the value the
    -- ISA's LOCAL_TENSOR range check and the CSR's capability register are
    -- expressed in -- is *derived* from these two, so the scratchpad and
    -- the address-space bound can never disagree.
    ----------------------------------------------------------------------
    g_num_banks : positive := 2;
    g_bank_words : positive := 1024;
    ----------------------------------------------------------------------
    -- Datapath bounds.
    ----------------------------------------------------------------------
    g_max_kernel_size : positive := cnn_accel_constant_max_kernel_size;
    -- Pooling's own, SEPARATE kernel bound. Larger than
    -- 'g_max_kernel_size' (5 vs 3) because YOLOv8n's SPPF block pools 5x5
    -- while every one of its convolutions is 1x1 or 3x3: sizing the
    -- shared bound to 5 would widen the PE array to 25 taps per lane and
    -- the weight buffer with it, for no benefit. Only the pool
    -- 'cnn_accel_window_gen' instance and the 'cnn_accel_pool' lane bank
    -- below are sized to this.
    g_max_pool_kernel_size : positive := cnn_accel_constant_max_pool_kernel_size;
    g_max_row_tile_words : positive := cnn_accel_constant_max_row_tile_words;
    g_accum_width : positive := cnn_accel_constant_accum_width;
    g_weight_buffer_depth : positive := cnn_accel_constant_weight_buffer_depth;
    g_bias_buffer_depth : positive := cnn_accel_constant_bias_buffer_depth;
    g_max_requant_shift : natural := 31;
    ----------------------------------------------------------------------
    -- External AXI4 geometry.
    ----------------------------------------------------------------------
    g_axi_addr_width : positive := 32;
    g_axi_data_width : positive := cnn_accel_constant_max_axi_data_width;
    -- Only used to size the DMA masters' internal ID plumbing. The read
    -- masters are serialized by 'cnn_accel_axi_mux' rather than distinguished
    -- by ID, so this value is a don't-care for correctness.
    g_axi_id_width : natural := 4;
    ----------------------------------------------------------------------
    -- Validation / liveness bounds, forwarded to 'cnn_accel_cmd_proc'.
    ----------------------------------------------------------------------
    g_ddr_limit : positive := 16#0020_0000#;
    g_watchdog_cycles : positive := 1_000_000
  );
  port (
    clk : in std_ulogic;
    -- Cold, synchronous active-high reset. See the reset note above.
    reset : in std_ulogic := '0';
    --# {{}}
    -- Host control/status: the generated register file, section 6.
    s_axi_lite_m2s : in axi_lite_m2s_t;
    s_axi_lite_s2m : out axi_lite_s2m_t := axi_lite_s2m_init;
    --# {{}}
    -- The accelerator's one and only DDR port: instruction fetch, tensor
    -- load, weight fill and store/spill, arbitrated by 'cnn_accel_axi_mux'.
    m_axi_m2s : out axi_m2s_t := axi_m2s_init;
    m_axi_s2m : in axi_s2m_t;
    --# {{}}
    -- Level interrupt, asserted while the CSR holds an unacknowledged
    -- done/error.
    irq : out std_ulogic := '0'
  );
end entity cnn_accel_top;

architecture a of cnn_accel_top is

  ------------------------------------------------------------------------
  -- Derived geometry.
  ------------------------------------------------------------------------

  constant c_word_bytes : positive := g_axi_data_width / 8;

  -- The LOCAL_TENSOR address space is exactly the scratchpad, so the bound
  -- used by validation (ERR_LOCAL_RANGE) and reported to the host is derived
  -- rather than declared -- spec section 3.
  constant c_tensor_bytes : positive := g_num_banks * g_bank_words * c_word_bytes;

  -- Maximum number of pooling taps in one window. Fixed by
  -- 'g_max_pool_kernel_size' alone: a pooling window is spatial only, the
  -- channel dimension is handled by the parallel lanes below.
  constant c_max_taps : positive := g_max_pool_kernel_size * g_max_pool_kernel_size;

  -- Read masters on the DDR port, in 'cnn_accel_axi_mux' input order.
  constant c_rd_instr : natural := 0;
  constant c_rd_load : natural := 1;
  constant c_rd_wgt : natural := 2;
  constant c_num_read_inputs : positive := 3;

  ------------------------------------------------------------------------
  -- Reset distribution.
  ------------------------------------------------------------------------

  -- Reaches the reset input of essentially every register in the design --
  -- a 22 674-load net in the first top-level build. Replicated rather than
  -- routed as one net; the semantics are unchanged (every replica is the
  -- same function of the same two sources, in the same cycle).
  signal reset_internal : std_ulogic := '0';
  attribute max_fanout : integer;
  attribute max_fanout of reset_internal : signal is 200;
  signal soft_reset_pulse : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- CSR <-> cmd_proc.
  ------------------------------------------------------------------------

  signal program_base_addr : std_ulogic_vector(g_axi_addr_width - 1 downto 0);
  signal start : std_ulogic := '0';
  signal seq_done : std_ulogic := '0';
  signal seq_error : std_ulogic := '0';
  signal err_code : std_ulogic_vector(3 downto 0) := (others => '0');
  signal err_pc : std_ulogic_vector(31 downto 0) := (others => '0');
  signal counters : csr_counters_t := csr_counters_init;

  ------------------------------------------------------------------------
  -- External AXI fan-in.
  ------------------------------------------------------------------------

  signal read_m2s_vec : axi_read_m2s_vec_t(0 to c_num_read_inputs - 1) :=
    (others => axi_read_m2s_init);
  signal read_s2m_vec : axi_read_s2m_vec_t(0 to c_num_read_inputs - 1);
  signal write_m2s_vec : axi_write_m2s_vec_t(0 to 0) := (others => axi_write_m2s_init);
  signal write_s2m_vec : axi_write_s2m_vec_t(0 to 0);

  signal axi_rd_bytes : unsigned(7 downto 0) := (others => '0');
  signal axi_wr_bytes : unsigned(7 downto 0) := (others => '0');

  -- Per-DMA halves of the vectors above. Kept as named signals (rather than
  -- port-mapping straight onto '<vec>(i).ar' etc.) so that every driver of a
  -- vector element is one concurrent aggregate assignment, which is what makes
  -- the fan-in readable in a netlist viewer.
  signal instr_ar_m2s, load_ar_m2s, wgt_ar_m2s : axi_m2s_a_t := axi_m2s_a_init;
  signal instr_ar_s2m, load_ar_s2m, wgt_ar_s2m : axi_s2m_a_t;
  signal instr_r_m2s, load_r_m2s, wgt_r_m2s : axi_m2s_r_t := axi_m2s_r_init;
  signal instr_r_s2m, load_r_s2m, wgt_r_s2m : axi_s2m_r_t;

  signal store_aw_m2s : axi_m2s_a_t := axi_m2s_a_init;
  signal store_aw_s2m : axi_s2m_a_t;
  signal store_w_m2s : axi_m2s_w_t := axi_m2s_w_init;
  signal store_w_s2m : axi_s2m_w_t;
  signal store_b_m2s : axi_m2s_b_t := axi_m2s_b_init;
  signal store_b_s2m : axi_s2m_b_t;

  ------------------------------------------------------------------------
  -- Instruction fetch.
  ------------------------------------------------------------------------

  signal fetch_start : std_ulogic := '0';
  signal fetch_addr : unsigned(31 downto 0) := (others => '0');
  signal fetch_desc : desc_v2_t := desc_v2_init;
  signal fetch_pc : unsigned(31 downto 0) := (others => '0');
  signal fetch_desc_valid : std_ulogic := '0';
  signal fetch_desc_ready : std_ulogic := '0';
  signal fetch_error : std_ulogic := '0';
  signal fetch_error_code : err_code_t := c_err_none;

  signal instr_req_m2s : dma_req_m2s_t;
  signal instr_req_s2m : dma_req_s2m_t;
  signal instr_dma_done : std_ulogic := '0';
  signal instr_resp_error : std_ulogic := '0';
  signal instr_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal instr_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  ------------------------------------------------------------------------
  -- DDR-side operand DMAs.
  ------------------------------------------------------------------------

  signal load_req_m2s : dma_req_m2s_t;
  signal load_req_s2m : dma_req_s2m_t;
  signal load_dma_done : std_ulogic := '0';
  signal load_resp_error : std_ulogic := '0';
  signal load_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal load_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  signal wgt_req_m2s : dma_req_m2s_t;
  signal wgt_req_s2m : dma_req_s2m_t;
  signal wgt_dma_done : std_ulogic := '0';
  signal wgt_resp_error : std_ulogic := '0';
  signal wgt_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal wgt_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  signal store_req_m2s : dma_req_m2s_t;
  signal store_req_s2m : dma_req_s2m_t;
  signal store_dma_done : std_ulogic := '0';
  signal store_resp_error : std_ulogic := '0';
  signal store_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal store_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  ------------------------------------------------------------------------
  -- Local tensor scratchpad ports.
  ------------------------------------------------------------------------

  ------------------------------------------------------------------------
  -- Registered request stage.
  --
  -- 'cnn_accel_cmd_proc's operand-port binding (its "who owns the source
  -- port" mux) selects, from a handful of its own state registers, which
  -- internal requester drives each physical request port. Feeding that mux
  -- straight into a sink's request-accept logic made one combinational
  -- path out of the selector register, the mux, the trip across the die,
  -- and the sink's own address decode or burst sizing -- the accelerator's
  -- worst remaining paths after the pool and window-generator reworks
  -- ('src0_is_ddr_q' into the scratchpad's read FSM, 'engine_conv_q' into
  -- the ifmap DMA's burst calculation).
  --
  -- Every request port therefore goes through a one-deep register stage
  -- below. Both directions are registered, so neither 'valid'/'addr' nor
  -- 'ready' crosses combinationally. The stage accepts one request every
  -- two cycles, which is free: a request is issued once per DMA job (a
  -- plane, a weight tile, a whole transfer), never per beat, and the
  -- issuing state machine waits for 'ready' anyway.
  ------------------------------------------------------------------------

  constant c_n_req_ports : positive := 7;
  type req_m2s_arr_t is array (0 to c_n_req_ports - 1) of dma_req_m2s_t;
  type req_s2m_arr_t is array (0 to c_n_req_ports - 1) of dma_req_s2m_t;

  -- Index into the two arrays below, one per physical request port.
  constant c_req_load : natural := 0;
  constant c_req_wgt : natural := 1;
  constant c_req_store : natural := 2;
  constant c_req_tm_w0 : natural := 3;
  constant c_req_tm_w1 : natural := 4;
  constant c_req_tm_r0 : natural := 5;
  constant c_req_tm_r1 : natural := 6;

  -- 'req_in_*': cmd_proc side. 'req_out_*': sink side.
  signal req_in_m2s : req_m2s_arr_t;
  signal req_in_s2m : req_s2m_arr_t;
  constant c_req_init : dma_req_m2s_t :=
    (valid => '0', req => (addr => (others => '0'), length => (others => '0')));
  signal req_out_m2s : req_m2s_arr_t := (others => c_req_init);
  signal req_out_s2m : req_s2m_arr_t;

  signal tm_w0_req_m2s, tm_w1_req_m2s : dma_req_m2s_t;
  signal tm_w0_req_s2m, tm_w1_req_s2m : dma_req_s2m_t;
  signal tm_w0_m2s, tm_w1_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  -- Scratchpad write streams after the skid stage below.
  signal tm_w0_piped_m2s, tm_w1_piped_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal tm_w0_piped_s2m, tm_w1_piped_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal tm_w0_s2m, tm_w1_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal tm_w0_done, tm_w1_done : std_ulogic := '0';

  signal tm_r0_req_m2s, tm_r1_req_m2s : dma_req_m2s_t;
  signal tm_r0_req_s2m, tm_r1_req_s2m : dma_req_s2m_t;
  signal tm_r0_m2s, tm_r1_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal tm_r0_s2m, tm_r1_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  -- Scratchpad read streams as the scratchpad itself drives them, i.e.
  -- upstream of the skid stage below.
  signal tm_r0_raw_m2s, tm_r1_raw_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal tm_r0_raw_s2m, tm_r1_raw_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal tm_r0_done, tm_r1_done : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Convolution engine.
  ------------------------------------------------------------------------

  signal conv_cfg_kernel_h, conv_cfg_kernel_w : std_ulogic_vector(7 downto 0);
  signal conv_cfg_stride_h, conv_cfg_stride_w : std_ulogic_vector(7 downto 0);
  signal conv_cfg_pad_top, conv_cfg_pad_bottom : std_ulogic_vector(7 downto 0);
  signal conv_cfg_pad_left, conv_cfg_pad_right : std_ulogic_vector(7 downto 0);
  -- ISA v2.1 'pad_value' for convolution -- see the conv_core port map.
  signal conv_cfg_pad_value : std_ulogic_vector(7 downto 0);
  signal conv_cfg_in_width, conv_cfg_in_height : std_ulogic_vector(15 downto 0);
  signal conv_cfg_in_channels : std_ulogic_vector(15 downto 0);
  signal conv_cfg_bias_en, conv_cfg_requant_en, conv_cfg_relu_en : std_ulogic;
  signal conv_cfg_requant_scale : std_ulogic_vector(31 downto 0);
  signal conv_cfg_requant_shift : std_ulogic_vector(7 downto 0);
  signal conv_cfg_output_offset : std_ulogic_vector(15 downto 0);
  signal conv_cfg_clamp_en : std_ulogic;
  signal conv_cfg_clamp_min, conv_cfg_clamp_max : std_ulogic_vector(7 downto 0);
  signal conv_cfg_per_channel_en : std_ulogic;
  -- Output frame dimensions, divided once per command by 'cnn_accel_cmd_proc'
  -- and latched by whichever window generator is started. Shared: only one
  -- engine runs per descriptor.
  signal geom_out_width, geom_out_height : std_ulogic_vector(15 downto 0);

  signal conv_start : std_ulogic := '0';
  signal conv_done : std_ulogic := '0';
  signal conv_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal conv_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal conv_fill_start, conv_fill_is_bias, conv_fill_is_scale : std_ulogic := '0';
  signal conv_weight_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal conv_weight_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal conv_out_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal conv_out_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  ------------------------------------------------------------------------
  -- Pool engine (see the "Pool engine" section in the body).
  ------------------------------------------------------------------------

  signal pool_cfg_kernel_h, pool_cfg_kernel_w : std_ulogic_vector(7 downto 0);
  signal pool_cfg_stride_h, pool_cfg_stride_w : std_ulogic_vector(7 downto 0);
  -- ISA v2.1 pooling padding: the four pad counts (already gated on
  -- FLAG_PAD_EN by 'cmd_proc') and the int8 value padded taps take.
  signal pool_cfg_pad_top, pool_cfg_pad_bottom : std_ulogic_vector(7 downto 0);
  signal pool_cfg_pad_left, pool_cfg_pad_right : std_ulogic_vector(7 downto 0);
  signal pool_cfg_pad_value : std_ulogic_vector(7 downto 0);
  signal pool_cfg_in_width, pool_cfg_in_height : std_ulogic_vector(15 downto 0);
  signal pool_cfg_opcode : std_ulogic_vector(7 downto 0);
  signal pool_cfg_requant_scale : std_ulogic_vector(31 downto 0);
  signal pool_cfg_requant_shift : std_ulogic_vector(7 downto 0);

  signal pool_start : std_ulogic := '0';
  signal pool_done : std_ulogic := '0';
  signal pool_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal pool_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  -- Same link, past the elastic stage below.
  signal pool_stream_piped_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal pool_stream_piped_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal pool_out_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal pool_out_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  signal pool_window_m2s :
    window_m2s_t(data(0 to window_data_length(g_max_pool_kernel_size, g_tile_channels) - 1));
  signal pool_window_s2m : window_s2m_t;

  -- One lane's ready, collected out of the per-lane generate below. The
  -- lane window records themselves are declared *inside* that generate
  -- (one scalar 'window_m2s_t' per lane) rather than as one vector signal
  -- here: 'window_m2s_t' has an unconstrained 'data' element, and an
  -- array of such a record would need a two-level element constraint that
  -- buys nothing over the per-lane declaration.
  signal pool_lane_ready_vec : std_ulogic_vector(0 to g_tile_channels - 1);
  signal pool_lane_max_m2s : axi_stream_m2s_vec_t(0 to g_tile_channels - 1);
  signal pool_lane_avgsum_m2s : axi_stream_m2s_vec_t(0 to g_tile_channels - 1);
  signal pool_max_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal pool_avgsum_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

  signal pool_max_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal pool_avg_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal pool_avg_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal pool_is_avg : std_ulogic := '0';

  signal pool_accum_m2s :
    accum_m2s_t(data(0 to g_tile_channels - 1)(g_accum_width - 1 downto 0));
  signal pool_accum_s2m : accum_s2m_t;

  ------------------------------------------------------------------------
  -- Elementwise engine.
  ------------------------------------------------------------------------

  signal ew_start : std_ulogic := '0';
  signal ew_opcode : std_ulogic_vector(7 downto 0);
  signal ew_src0_addr, ew_src1_addr, ew_dst_addr, ew_lut_addr : unsigned(31 downto 0);
  signal ew_xfer_bytes : unsigned(31 downto 0);
  signal ew_in_width, ew_in_height, ew_in_channels : unsigned(15 downto 0);
  signal ew_requant_scale : signed(31 downto 0);
  signal ew_requant_shift : unsigned(7 downto 0);
  signal ew_done : std_ulogic := '0';
  signal ew_error : std_ulogic := '0';
  signal ew_error_code : err_code_t := c_err_none;

  signal ew_src0_req_m2s, ew_src1_req_m2s : dma_req_m2s_t;
  signal ew_src0_req_s2m, ew_src1_req_s2m : dma_req_s2m_t;
  signal ew_dst_req_m2s, ew_lut_req_m2s : dma_req_m2s_t;
  signal ew_dst_req_s2m, ew_lut_req_s2m : dma_req_s2m_t;
  signal ew_src0_stream_m2s, ew_src1_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal ew_src0_stream_s2m, ew_src1_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal ew_dst_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal ew_dst_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;
  signal ew_lut_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal ew_lut_stream_s2m : axi_stream_s2m_t := axi_stream_s2m_init;

begin

  ------------------------------------------------------------------------
  -- Geometry contracts. These are the places where a top-level generic
  -- override could silently produce a design that computes the wrong thing
  -- rather than one that fails to build, so each is checked here even though
  -- some of the sub-blocks repeat the check.
  ------------------------------------------------------------------------

  assert g_tile_channels = cnn_accel_constant_activation_plane_channels
    report "cnn_accel_top: g_tile_channels (" & positive'image(g_tile_channels) &
      ") must equal the generated activation-plane channel count (" &
      integer'image(cnn_accel_constant_activation_plane_channels) &
      "): one scratchpad/DDR activation word is exactly one input-channel tile"
    severity failure;

  assert g_pe_rows = cnn_accel_constant_activation_plane_channels
    report "cnn_accel_top: g_pe_rows (" & positive'image(g_pe_rows) &
      ") must equal the generated activation-plane channel count (" &
      integer'image(cnn_accel_constant_activation_plane_channels) &
      "): one OT pass writes exactly one output activation plane"
    severity failure;

  -- The pool lane window used to have to fit the fixed 128-bit
  -- 'axi_stream_m2s_t.data' ('c_max_taps * 8 <= axi_stream_data_sz'),
  -- which capped the pool kernel at 3 (9 taps = 72 bits; 5x5 would be 200
  -- bits). It no longer travels in an AXI-Stream record at all: each lane
  -- gets its own unconstrained 'window_m2s_t' tap array, the same record
  -- type the conv path already uses, which has no width ceiling. Widening
  -- the hdl-modules-wide 'axi_stream_data_sz' instead was rejected -- it
  -- would inflate every stream in the design for one consumer's benefit.
  --
  -- What remains checkable here is that pooling is genuinely the *only*
  -- consumer sized to the larger bound, i.e. that raising
  -- 'g_max_pool_kernel_size' cannot silently drag the conv datapath along.
  assert g_max_pool_kernel_size >= g_max_kernel_size
    report "cnn_accel_top: g_max_pool_kernel_size (" &
      positive'image(g_max_pool_kernel_size) & ") must be >= g_max_kernel_size (" &
      positive'image(g_max_kernel_size) &
      "): pooling reuses the conv path's window generator geometry bounds"
    severity failure;

  assert g_axi_data_width = 8 * g_tile_channels
    report "cnn_accel_top: g_axi_data_width (" & positive'image(g_axi_data_width) &
      ") must be 8 * g_tile_channels -- one bus word is one activation tile"
    severity failure;

  ------------------------------------------------------------------------
  -- Reset. 'soft_reset_pulse' is a single-cycle CSR write side effect; ORing
  -- it into the datapath reset is exactly the "abort at any point" semantics
  -- the spec asks for, and it costs nothing because every block already has a
  -- synchronous reset input.
  ------------------------------------------------------------------------

  reset_internal <= reset or soft_reset_pulse;

  ------------------------------------------------------------------------
  -- Host control/status.
  ------------------------------------------------------------------------

  csr_inst : entity cnn_accel.cnn_accel_csr
    generic map (
      g_pe_rows => g_pe_rows,
      g_pe_cols => g_pe_cols,
      g_tile_channels => g_tile_channels,
      g_max_kernel_size => g_max_kernel_size,
      g_max_pool_kernel_size => g_max_pool_kernel_size,
      g_max_row_tile_words => g_max_row_tile_words,
      g_tensor_bytes => c_tensor_bytes,
      g_axi_addr_width => g_axi_addr_width
    )
    port map (
      clk => clk,
      -- Deliberately the cold reset: a soft reset must abort the datapath
      -- without destroying the status the host is about to read back.
      reset => reset,

      s_axi_lite_m2s => s_axi_lite_m2s,
      s_axi_lite_s2m => s_axi_lite_s2m,

      program_base_addr => program_base_addr,
      start => start,
      soft_reset_pulse => soft_reset_pulse,

      seq_done => seq_done,
      seq_error => seq_error,
      err_code => err_code,
      err_pc => err_pc,

      counters => counters,

      irq => irq
    );

  ------------------------------------------------------------------------
  -- Command processor: the only block with an opinion about the ISA.
  ------------------------------------------------------------------------

  cmd_proc_inst : entity cnn_accel.cnn_accel_cmd_proc
    generic map (
      g_pe_rows => g_pe_rows,
      g_pe_cols => g_pe_cols,
      g_tile_channels => g_tile_channels,
      g_max_kernel_size => g_max_kernel_size,
      g_max_pool_kernel_size => g_max_pool_kernel_size,
      g_max_row_tile_words => g_max_row_tile_words,
      g_tensor_bytes => c_tensor_bytes,
      g_ddr_limit => g_ddr_limit,
      g_watchdog_cycles => g_watchdog_cycles
    )
    port map (
      clk => clk,
      reset => reset,

      start => start,
      program_base_addr => program_base_addr,
      soft_reset_pulse => soft_reset_pulse,
      seq_done => seq_done,
      seq_error => seq_error,
      err_code => err_code,
      err_pc => err_pc,
      counters => counters,

      axi_rd_bytes => axi_rd_bytes,
      axi_wr_bytes => axi_wr_bytes,

      fetch_start => fetch_start,
      fetch_addr => fetch_addr,
      fetch_desc => fetch_desc,
      fetch_pc => fetch_pc,
      fetch_desc_valid => fetch_desc_valid,
      fetch_desc_ready => fetch_desc_ready,
      fetch_error => fetch_error,
      fetch_error_code => fetch_error_code,

      load_req_m2s => load_req_m2s,
      load_req_s2m => load_req_s2m,
      load_dma_done => load_dma_done,
      load_resp_error => load_resp_error,
      s_load_stream_m2s => load_stream_m2s,
      s_load_stream_s2m => load_stream_s2m,

      wgt_req_m2s => wgt_req_m2s,
      wgt_req_s2m => wgt_req_s2m,
      wgt_dma_done => wgt_dma_done,
      wgt_resp_error => wgt_resp_error,
      s_wgt_stream_m2s => wgt_stream_m2s,
      s_wgt_stream_s2m => wgt_stream_s2m,

      store_req_m2s => store_req_m2s,
      store_req_s2m => store_req_s2m,
      store_dma_done => store_dma_done,
      store_resp_error => store_resp_error,
      m_store_stream_m2s => store_stream_m2s,
      m_store_stream_s2m => store_stream_s2m,

      tm_w0_req_m2s => tm_w0_req_m2s,
      tm_w0_req_s2m => tm_w0_req_s2m,
      m_tm_w0_m2s => tm_w0_m2s,
      m_tm_w0_s2m => tm_w0_s2m,
      tm_w0_done => tm_w0_done,

      tm_w1_req_m2s => tm_w1_req_m2s,
      tm_w1_req_s2m => tm_w1_req_s2m,
      m_tm_w1_m2s => tm_w1_m2s,
      m_tm_w1_s2m => tm_w1_s2m,
      tm_w1_done => tm_w1_done,

      tm_r0_req_m2s => tm_r0_req_m2s,
      tm_r0_req_s2m => tm_r0_req_s2m,
      s_tm_r0_m2s => tm_r0_m2s,
      s_tm_r0_s2m => tm_r0_s2m,
      tm_r0_done => tm_r0_done,

      tm_r1_req_m2s => tm_r1_req_m2s,
      tm_r1_req_s2m => tm_r1_req_s2m,
      s_tm_r1_m2s => tm_r1_m2s,
      s_tm_r1_s2m => tm_r1_s2m,
      tm_r1_done => tm_r1_done,

      conv_cfg_kernel_h => conv_cfg_kernel_h,
      conv_cfg_kernel_w => conv_cfg_kernel_w,
      conv_cfg_stride_h => conv_cfg_stride_h,
      conv_cfg_stride_w => conv_cfg_stride_w,
      conv_cfg_pad_top => conv_cfg_pad_top,
      conv_cfg_pad_bottom => conv_cfg_pad_bottom,
      conv_cfg_pad_left => conv_cfg_pad_left,
      conv_cfg_pad_right => conv_cfg_pad_right,
      conv_cfg_pad_value => conv_cfg_pad_value,
      conv_cfg_in_width => conv_cfg_in_width,
      conv_cfg_in_height => conv_cfg_in_height,
      conv_cfg_in_channels => conv_cfg_in_channels,
      conv_cfg_bias_en => conv_cfg_bias_en,
      conv_cfg_requant_en => conv_cfg_requant_en,
      conv_cfg_relu_en => conv_cfg_relu_en,
      conv_cfg_requant_scale => conv_cfg_requant_scale,
      conv_cfg_requant_shift => conv_cfg_requant_shift,
      conv_cfg_output_offset => conv_cfg_output_offset,
      conv_cfg_clamp_en => conv_cfg_clamp_en,
      conv_cfg_clamp_min => conv_cfg_clamp_min,
      conv_cfg_clamp_max => conv_cfg_clamp_max,
      conv_cfg_per_channel_en => conv_cfg_per_channel_en,
      geom_out_width => geom_out_width,
      geom_out_height => geom_out_height,
      conv_start => conv_start,
      conv_done => conv_done,
      m_conv_stream_m2s => conv_stream_m2s,
      m_conv_stream_s2m => conv_stream_s2m,
      conv_fill_start => conv_fill_start,
      conv_fill_is_bias => conv_fill_is_bias,
      conv_fill_is_scale => conv_fill_is_scale,
      m_conv_weight_m2s => conv_weight_m2s,
      m_conv_weight_s2m => conv_weight_s2m,
      s_conv_out_m2s => conv_out_m2s,
      s_conv_out_s2m => conv_out_s2m,

      pool_cfg_kernel_h => pool_cfg_kernel_h,
      pool_cfg_kernel_w => pool_cfg_kernel_w,
      pool_cfg_stride_h => pool_cfg_stride_h,
      pool_cfg_stride_w => pool_cfg_stride_w,
      pool_cfg_pad_top => pool_cfg_pad_top,
      pool_cfg_pad_bottom => pool_cfg_pad_bottom,
      pool_cfg_pad_left => pool_cfg_pad_left,
      pool_cfg_pad_right => pool_cfg_pad_right,
      pool_cfg_pad_value => pool_cfg_pad_value,
      pool_cfg_in_width => pool_cfg_in_width,
      pool_cfg_in_height => pool_cfg_in_height,
      pool_cfg_opcode => pool_cfg_opcode,
      pool_cfg_requant_scale => pool_cfg_requant_scale,
      pool_cfg_requant_shift => pool_cfg_requant_shift,
      pool_start => pool_start,
      pool_done => pool_done,
      m_pool_stream_m2s => pool_stream_m2s,
      m_pool_stream_s2m => pool_stream_s2m,
      s_pool_out_m2s => pool_out_m2s,
      s_pool_out_s2m => pool_out_s2m,

      ew_start => ew_start,
      ew_opcode => ew_opcode,
      ew_src0_addr => ew_src0_addr,
      ew_src1_addr => ew_src1_addr,
      ew_dst_addr => ew_dst_addr,
      ew_lut_addr => ew_lut_addr,
      ew_xfer_bytes => ew_xfer_bytes,
      ew_in_width => ew_in_width,
      ew_in_height => ew_in_height,
      ew_in_channels => ew_in_channels,
      ew_requant_scale => ew_requant_scale,
      ew_requant_shift => ew_requant_shift,
      ew_done => ew_done,
      ew_error => ew_error,
      ew_error_code => ew_error_code,

      ew_src0_req_m2s => ew_src0_req_m2s,
      ew_src0_req_s2m => ew_src0_req_s2m,
      m_ew_src0_stream_m2s => ew_src0_stream_m2s,
      m_ew_src0_stream_s2m => ew_src0_stream_s2m,
      ew_src1_req_m2s => ew_src1_req_m2s,
      ew_src1_req_s2m => ew_src1_req_s2m,
      m_ew_src1_stream_m2s => ew_src1_stream_m2s,
      m_ew_src1_stream_s2m => ew_src1_stream_s2m,
      ew_dst_req_m2s => ew_dst_req_m2s,
      ew_dst_req_s2m => ew_dst_req_s2m,
      s_ew_dst_stream_m2s => ew_dst_stream_m2s,
      s_ew_dst_stream_s2m => ew_dst_stream_s2m,
      ew_lut_req_m2s => ew_lut_req_m2s,
      ew_lut_req_s2m => ew_lut_req_s2m,
      m_ew_lut_stream_m2s => ew_lut_stream_m2s,
      m_ew_lut_stream_s2m => ew_lut_stream_s2m
    );

  ------------------------------------------------------------------------
  -- Instruction fetch: 64-byte descriptor assembler on its own DDR read
  -- master, so an instruction fetch never has to wait behind a tensor load.
  -- 'g_timeout_cycles' is tied to the same watchdog bound as 'cmd_proc' so
  -- that a DDR that never responds surfaces as ERR_TIMEOUT from whichever of
  -- the two notices first.
  ------------------------------------------------------------------------

  cmd_fetch_inst : entity cnn_accel.cnn_accel_cmd_fetch
    generic map (
      g_axi_data_width => g_axi_data_width,
      g_timeout_cycles => g_watchdog_cycles
    )
    port map (
      clk => clk,
      reset => reset_internal,

      start => fetch_start,
      addr => fetch_addr,

      instr_req_m2s => instr_req_m2s,
      instr_req_s2m => instr_req_s2m,

      s_instr_stream_m2s => instr_stream_m2s,
      s_instr_stream_s2m => instr_stream_s2m,

      instr_dma_done => instr_dma_done,
      instr_resp_error => instr_resp_error,

      desc => fetch_desc,
      pc => fetch_pc,
      desc_valid => fetch_desc_valid,
      desc_ready => fetch_desc_ready,

      error => fetch_error,
      error_code => fetch_error_code
    );

  ------------------------------------------------------------------------
  -- DDR read masters. Three instances of the same block, differing only in
  -- who owns the request port: instruction fetch, tensor/activation load, and
  -- the weight/bias/scale/side operand.
  ------------------------------------------------------------------------

  instr_read_dma_inst : entity cnn_accel.cnn_accel_axi_read_dma
    generic map (
      g_axi_addr_width => g_axi_addr_width,
      g_axi_data_width => g_axi_data_width,
      g_axi_id_width => g_axi_id_width
    )
    port map (
      clk => clk,
      reset => reset_internal,

      req_m2s => instr_req_m2s,
      req_s2m => instr_req_s2m,
      dma_done => instr_dma_done,
      resp_error => instr_resp_error,

      m_axi_ar_m2s => instr_ar_m2s,
      m_axi_ar_s2m => instr_ar_s2m,
      m_axi_r_m2s => instr_r_m2s,
      m_axi_r_s2m => instr_r_s2m,

      m_stream_m2s => instr_stream_m2s,
      m_stream_s2m => instr_stream_s2m
    );

  load_read_dma_inst : entity cnn_accel.cnn_accel_axi_read_dma
    generic map (
      g_axi_addr_width => g_axi_addr_width,
      g_axi_data_width => g_axi_data_width,
      g_axi_id_width => g_axi_id_width
    )
    port map (
      clk => clk,
      reset => reset_internal,

      req_m2s => req_out_m2s(c_req_load),
      req_s2m => req_out_s2m(c_req_load),
      dma_done => load_dma_done,
      resp_error => load_resp_error,

      m_axi_ar_m2s => load_ar_m2s,
      m_axi_ar_s2m => load_ar_s2m,
      m_axi_r_m2s => load_r_m2s,
      m_axi_r_s2m => load_r_s2m,

      m_stream_m2s => load_stream_m2s,
      m_stream_s2m => load_stream_s2m
    );

  wgt_read_dma_inst : entity cnn_accel.cnn_accel_axi_read_dma
    generic map (
      g_axi_addr_width => g_axi_addr_width,
      g_axi_data_width => g_axi_data_width,
      g_axi_id_width => g_axi_id_width
    )
    port map (
      clk => clk,
      reset => reset_internal,

      req_m2s => req_out_m2s(c_req_wgt),
      req_s2m => req_out_s2m(c_req_wgt),
      dma_done => wgt_dma_done,
      resp_error => wgt_resp_error,

      m_axi_ar_m2s => wgt_ar_m2s,
      m_axi_ar_s2m => wgt_ar_s2m,
      m_axi_r_m2s => wgt_r_m2s,
      m_axi_r_s2m => wgt_r_s2m,

      m_stream_m2s => wgt_stream_m2s,
      m_stream_s2m => wgt_stream_s2m
    );

  ------------------------------------------------------------------------
  -- The one DDR write master: STORE and spill.
  ------------------------------------------------------------------------

  ofmap_dma_inst : entity cnn_accel.cnn_accel_ofmap_dma
    generic map (
      g_axi_addr_width => g_axi_addr_width,
      g_axi_data_width => g_axi_data_width
    )
    port map (
      clk => clk,
      reset => reset_internal,

      req_m2s => req_out_m2s(c_req_store),
      req_s2m => req_out_s2m(c_req_store),
      dma_done => store_dma_done,
      resp_error => store_resp_error,

      s_stream_m2s => store_stream_m2s,
      s_stream_s2m => store_stream_s2m,

      m_axi_aw_m2s => store_aw_m2s,
      m_axi_aw_s2m => store_aw_s2m,
      m_axi_w_m2s => store_w_m2s,
      m_axi_w_s2m => store_w_s2m,
      m_axi_b_m2s => store_b_m2s,
      m_axi_b_s2m => store_b_s2m
    );

  ------------------------------------------------------------------------
  -- DDR port arbitration and traffic measurement. The byte increments feed
  -- 'cmd_proc's DDR_RD_BYTES/DDR_WR_BYTES counters, which is why they are
  -- taken here -- at the single point every DDR beat must pass through --
  -- rather than estimated from descriptor lengths.
  ------------------------------------------------------------------------

  read_m2s_vec(c_rd_instr) <= (ar => instr_ar_m2s, r => instr_r_m2s);
  read_m2s_vec(c_rd_load) <= (ar => load_ar_m2s, r => load_r_m2s);
  read_m2s_vec(c_rd_wgt) <= (ar => wgt_ar_m2s, r => wgt_r_m2s);

  instr_ar_s2m <= read_s2m_vec(c_rd_instr).ar;
  instr_r_s2m <= read_s2m_vec(c_rd_instr).r;
  load_ar_s2m <= read_s2m_vec(c_rd_load).ar;
  load_r_s2m <= read_s2m_vec(c_rd_load).r;
  wgt_ar_s2m <= read_s2m_vec(c_rd_wgt).ar;
  wgt_r_s2m <= read_s2m_vec(c_rd_wgt).r;

  write_m2s_vec(0) <= (aw => store_aw_m2s, w => store_w_m2s, b => store_b_m2s);

  store_aw_s2m <= write_s2m_vec(0).aw;
  store_w_s2m <= write_s2m_vec(0).w;
  store_b_s2m <= write_s2m_vec(0).b;

  axi_mux_inst : entity cnn_accel.cnn_accel_axi_mux
    generic map (
      g_axi_data_width => g_axi_data_width,
      g_num_read_inputs => c_num_read_inputs,
      g_num_write_inputs => 1
    )
    port map (
      clk => clk,

      input_read_m2s => read_m2s_vec,
      input_read_s2m => read_s2m_vec,

      input_write_m2s => write_m2s_vec,
      input_write_s2m => write_s2m_vec,

      m_axi_m2s => m_axi_m2s,
      m_axi_s2m => m_axi_s2m,

      rd_bytes => axi_rd_bytes,
      wr_bytes => axi_wr_bytes
    );

  ------------------------------------------------------------------------
  -- Local tensor scratchpad. Two write and two read channels, matching the
  -- worst-case simultaneous demand of one command: an engine reading its
  -- source and its side operand while writing its destination, plus the
  -- separate pure-move write port.
  ------------------------------------------------------------------------

  tensor_mem_inst : entity cnn_accel.cnn_accel_tensor_mem
    generic map (
      g_num_banks => g_num_banks,
      g_bank_words => g_bank_words,
      g_data_width => g_axi_data_width
    )
    port map (
      clk => clk,
      reset => reset_internal,

      w0_req_m2s => req_out_m2s(c_req_tm_w0),
      w0_req_s2m => req_out_s2m(c_req_tm_w0),
      s_w0_m2s => tm_w0_piped_m2s,
      s_w0_s2m => tm_w0_piped_s2m,
      w0_done => tm_w0_done,

      w1_req_m2s => req_out_m2s(c_req_tm_w1),
      w1_req_s2m => req_out_s2m(c_req_tm_w1),
      s_w1_m2s => tm_w1_piped_m2s,
      s_w1_s2m => tm_w1_piped_s2m,
      w1_done => tm_w1_done,

      r0_req_m2s => req_out_m2s(c_req_tm_r0),
      r0_req_s2m => req_out_s2m(c_req_tm_r0),
      m_r0_m2s => tm_r0_raw_m2s,
      m_r0_s2m => tm_r0_raw_s2m,

      r1_req_m2s => req_out_m2s(c_req_tm_r1),
      r1_req_s2m => req_out_s2m(c_req_tm_r1),
      m_r1_m2s => tm_r1_raw_m2s,
      m_r1_s2m => tm_r1_raw_s2m,
      -- Both read-channel 'done' pulses are re-derived below, on the far
      -- side of the skid stage. 'cnn_accel_tensor_mem' raises its own
      -- when the last beat is accepted by whatever is connected to
      -- 'm_rN_s2m.ready' -- which, since the skid stage was inserted, is
      -- the pipeline register and not the consumer. See
      -- 'tm_read_pipeline_gen'.
      r0_done => open,
      r1_done => open
    );

  ------------------------------------------------------------------------
  -- Convolution engine (CONV2D / FC). Self-contained: it brings its own
  -- window_gen, pe_array, weight_buffer and the fused bias/requant/clamp
  -- epilogue, so the fusion rule of spec section 5.3 is satisfied
  -- structurally -- there is no wire here on which an int32 tensor could be
  -- materialized.
  ------------------------------------------------------------------------

  conv_core_inst : entity cnn_accel.cnn_accel_conv_core
    generic map (
      g_pe_rows => g_pe_rows,
      g_pe_cols => g_pe_cols,
      g_accum_width => g_accum_width,
      g_max_kernel_size => g_max_kernel_size,
      g_tile_channels => g_tile_channels,
      g_max_row_tile_words => g_max_row_tile_words,
      g_weight_buffer_depth => g_weight_buffer_depth,
      g_bias_buffer_depth => g_bias_buffer_depth,
      g_max_requant_shift => g_max_requant_shift
    )
    port map (
      clk => clk,
      reset => reset_internal,

      cfg_kernel_h => conv_cfg_kernel_h,
      cfg_kernel_w => conv_cfg_kernel_w,
      cfg_stride_h => conv_cfg_stride_h,
      cfg_stride_w => conv_cfg_stride_w,
      cfg_pad_top => conv_cfg_pad_top,
      cfg_pad_bottom => conv_cfg_pad_bottom,
      cfg_pad_left => conv_cfg_pad_left,
      cfg_pad_right => conv_cfg_pad_right,
      -- ISA v2.1: convolution pads with the descriptor's 'pad_value', not
      -- with a literal 0. For an int8 tensor with a nonzero zero-point the
      -- two differ by a constant on every padded tap, and a 3x3/pad-1
      -- convolution (YOLOv8n's shape throughout) pads every border output
      -- of every layer, so this is an accuracy bug and not a rounding
      -- artefact. 'cmd_proc' passes the field straight through; a
      -- descriptor that leaves it 0 zero-pads exactly as before.
      cfg_pad_value => conv_cfg_pad_value,
      cfg_in_width => conv_cfg_in_width,
      cfg_in_height => conv_cfg_in_height,
      cfg_in_channels => conv_cfg_in_channels,

      cfg_bias_en => conv_cfg_bias_en,
      cfg_requant_en => conv_cfg_requant_en,
      cfg_relu_en => conv_cfg_relu_en,
      cfg_requant_scale => conv_cfg_requant_scale,
      cfg_requant_shift => conv_cfg_requant_shift,
      cfg_output_offset => conv_cfg_output_offset,
      cfg_clamp_en => conv_cfg_clamp_en,
      cfg_clamp_min => conv_cfg_clamp_min,
      cfg_clamp_max => conv_cfg_clamp_max,
      cfg_per_channel_en => conv_cfg_per_channel_en,
      cfg_out_width => geom_out_width,
      cfg_out_height => geom_out_height,

      start => conv_start,
      done => conv_done,

      s_stream_m2s => conv_stream_m2s,
      s_stream_s2m => conv_stream_s2m,

      fill_start => conv_fill_start,
      fill_is_bias => conv_fill_is_bias,
      fill_is_scale => conv_fill_is_scale,
      s_weight_m2s => conv_weight_m2s,
      s_weight_s2m => conv_weight_s2m,

      m_out_m2s => conv_out_m2s,
      m_out_s2m => conv_out_s2m
    );

  ------------------------------------------------------------------------
  -- Pool engine (POOL_MAX / POOL_AVG).
  --
  -- The architecture document names the pieces -- "window_gen + pool
  -- (+ bias_requant)" -- but rev 1's 'cnn_accel_layer_ctrl', which was to own
  -- the glue between them, was never built, so the assembly is defined here.
  -- Two interface facts drive the shape:
  --
  --  * 'cnn_accel_pool' reduces ONE channel's window per beat: its
  --    's_window_m2s.data' holds tap 'i' at bits '8i+7 downto 8i' and it emits
  --    one int8 ('m_max') or one int32 ('m_avgsum').
  --  * 'cnn_accel_window_gen' emits 'data(i * g_tile_channels + c)' = tap 'i'
  --    of channel 'c' -- a whole activation tile per beat.
  --
  -- So one 'window_gen' feeds 'g_tile_channels' pool lanes in lockstep, each
  -- lane getting a static stride-'g_tile_channels' slice of the window. The
  -- lanes are identical combinational-plus-one-register reducers driven by the
  -- same 'valid'/'ready', so they never diverge; the shared handshake is the
  -- AND of the lanes' readys.
  --
  -- That width is not an arbitrary choice: it is what makes the two outputs
  -- land on the interfaces the reused blocks already have. The MAX lanes pack
  -- straight into one activation word, and the AVG lanes form exactly the
  -- 'g_pe_rows'-lane 'accum_m2s_t' that 'cnn_accel_bias_requant' consumes, so
  -- the pool-area divide is the same requantizer the conv epilogue uses, with
  -- bias and per-channel scaling switched off.
  --
  -- Channel tiling is handled upstream: 'cmd_proc' runs pooling as one pass
  -- per activation plane, so within a pass 'in_channels' is always exactly one
  -- tile and the window_gen's tile count is 1. That is why 'cfg_in_channels'
  -- is a constant here and why the output planes come out in
  -- '[tile][y][x]' order without any output-side reordering.
  ------------------------------------------------------------------------

  pool_window_gen_inst : entity cnn_accel.cnn_accel_window_gen
    generic map (
      -- The POOL bound, not the conv one: this instance is what a 5x5
      -- SPPF pool needs, and it is the only place that pays for it.
      g_max_kernel_size => g_max_pool_kernel_size,
      g_max_row_tile_words => g_max_row_tile_words,
      g_tile_channels => g_tile_channels
    )
    port map (
      clk => clk,
      reset => reset_internal,

      cfg_kernel_h => pool_cfg_kernel_h,
      cfg_kernel_w => pool_cfg_kernel_w,
      cfg_stride_h => pool_cfg_stride_h,
      cfg_stride_w => pool_cfg_stride_w,
      -- ISA v2.1: pooling has padding. It reuses CONV2D's own
      -- 'FLAG_PAD_EN' + pad_top/bottom/left/right fields ('cmd_proc'
      -- gates them on the flag and zeroes them otherwise, exactly as it
      -- does for conv), and 'cfg_pad_value' -- the int8 value a padded
      -- tap takes. That value is the input tensor's quantization
      -- zero-point, NOT 0: a 0 tap in a max pool over a zero_point =
      -- -128 tensor is larger than nearly every real activation and
      -- wins every border output. YOLOv8n's SPPF (5x5, stride 1, pad 2)
      -- is the shape this exists for.
      cfg_pad_top => pool_cfg_pad_top,
      cfg_pad_bottom => pool_cfg_pad_bottom,
      cfg_pad_left => pool_cfg_pad_left,
      cfg_pad_right => pool_cfg_pad_right,
      cfg_pad_value => pool_cfg_pad_value,
      cfg_in_width => pool_cfg_in_width,
      cfg_in_height => pool_cfg_in_height,
      cfg_in_channels => std_ulogic_vector(to_unsigned(g_tile_channels, 16)),
      cfg_out_width => geom_out_width,
      cfg_out_height => geom_out_height,

      start => pool_start,
      -- The engine's completion is taken from the far end of the pipeline
      -- (below), not from the feeder, so that the last window is known to have
      -- been reduced and accepted.
      done => open,

      s_stream_m2s => pool_stream_piped_m2s,
      s_stream_s2m => pool_stream_piped_s2m,

      m_window_m2s => pool_window_m2s,
      m_window_s2m => pool_window_s2m
    );

  ------------------------------------------------------------------------
  -- Elastic stage on the POOL lane's activation ingest.
  --
  -- The CONV lane already has exactly this stage: 'cnn_accel_conv_core'
  -- instantiates 's_stream_pipeline_inst' on its own 's_stream' before
  -- handing it to its window generator. The pool lane's window generator is
  -- instantiated directly here and had none, so its 's_stream_s2m.ready' --
  -- a combinational function of the reservation counter 'n_res_q' and two
  -- CARRY4s -- ran straight back into 'cnn_accel_cmd_proc's 'src_ready'
  -- mux, from there into the physical read ports' back-pressure, the feeder
  -- FSM, the performance counters and the watchdog reload. Post-route that
  -- single net owned 176 of the design's worst 400 endpoints at 10-11 logic
  -- levels and 71 % route delay.
  --
  -- 'shared/TimingAndResources.md' section 2, "A ready chain across several
  -- modules is the same failure with the fix at the wrong end": the fix is a
  -- skid/elastic register at the hierarchy boundary, not placement. This is
  -- the same 'common.handshake_pipeline' idiom, with the same generics, as
  -- 'conv_core's stage and as 'tm_read_pipeline_gen' below --
  -- 'full_throughput => true', so the lane still sustains one beat per
  -- cycle and only its latency moves, by one cycle.
  --
  -- Unlike 'tm_read_pipeline_gen' there is no 'done' to re-derive here: the
  -- pool engine's completion is already taken from the far end of the
  -- reduction pipeline ('pool_done', see the 'done => open' above), never
  -- from beats accepted on this link.
  ------------------------------------------------------------------------

  pool_stream_pipeline_inst : entity common.handshake_pipeline
    generic map (
      data_width => axi_stream_data_sz,
      full_throughput => true,
      pipeline_control_signals => true,
      pipeline_data_signals => true
    )
    port map (
      clk => clk,

      input_ready => pool_stream_s2m.ready,
      input_valid => pool_stream_m2s.valid,
      input_last => pool_stream_m2s.last,
      input_data => pool_stream_m2s.data,

      output_ready => pool_stream_piped_s2m.ready,
      output_valid => pool_stream_piped_m2s.valid,
      output_last => pool_stream_piped_m2s.last,
      output_data => pool_stream_piped_m2s.data
    );

  pool_stream_piped_m2s.user <= (others => '-');

  req_in_m2s(c_req_load) <= load_req_m2s;
  req_in_m2s(c_req_wgt) <= wgt_req_m2s;
  req_in_m2s(c_req_store) <= store_req_m2s;
  req_in_m2s(c_req_tm_w0) <= tm_w0_req_m2s;
  req_in_m2s(c_req_tm_w1) <= tm_w1_req_m2s;
  req_in_m2s(c_req_tm_r0) <= tm_r0_req_m2s;
  req_in_m2s(c_req_tm_r1) <= tm_r1_req_m2s;

  load_req_s2m <= req_in_s2m(c_req_load);
  wgt_req_s2m <= req_in_s2m(c_req_wgt);
  store_req_s2m <= req_in_s2m(c_req_store);
  tm_w0_req_s2m <= req_in_s2m(c_req_tm_w0);
  tm_w1_req_s2m <= req_in_s2m(c_req_tm_w1);
  tm_r0_req_s2m <= req_in_s2m(c_req_tm_r0);
  tm_r1_req_s2m <= req_in_s2m(c_req_tm_r1);

  req_pipeline : process(clk)
  begin
    if rising_edge(clk) then
      for i in 0 to c_n_req_ports - 1 loop
        if req_out_m2s(i).valid = '0' then
          if req_in_m2s(i).valid = '1' then
            req_out_m2s(i) <= req_in_m2s(i);
          end if;
        elsif req_out_s2m(i).ready = '1' then
          req_out_m2s(i).valid <= '0';
        end if;
      end loop;

      if reset_internal = '1' then
        for i in 0 to c_n_req_ports - 1 loop
          req_out_m2s(i).valid <= '0';
        end loop;
      end if;
    end if;
  end process;

  -- Registered, so nothing downstream of the stage reaches cmd_proc's mux.
  req_ready_gen : for i in 0 to c_n_req_ports - 1 generate
    req_in_s2m(i).ready <= not req_out_m2s(i).valid;
  end generate;

  ------------------------------------------------------------------------
  -- Skid stage on each scratchpad read stream.
  --
  -- The mirror image of the write-side stage below, and it exists for the
  -- mirror-image path: the ifmap consumer's 'ready' -- the window
  -- generator's, ultimately its tap-assembly launch control -- reached
  -- 'cnn_accel_tensor_mem's block-RAM read ADDRESS through
  -- 'cnn_accel_cmd_proc's operand mux, one combinational path from a
  -- window-generator register to a 'ADDRBWRADDR' pin.
  ------------------------------------------------------------------------

  tm_read_pipeline_gen : for r in 0 to 1 generate
    signal input_m2s, output_m2s : axi_stream_m2s_t;
    signal input_ready, output_ready, output_done : std_ulogic;
  begin
    input_m2s <= tm_r0_raw_m2s when r = 0 else tm_r1_raw_m2s;
    output_ready <= tm_r0_s2m.ready when r = 0 else tm_r1_s2m.ready;

    pipeline_inst : entity common.handshake_pipeline
      generic map (
        data_width => axi_stream_data_sz,
        full_throughput => true,
        pipeline_control_signals => true,
        pipeline_data_signals => true
      )
      port map (
        clk => clk,

        input_ready => input_ready,
        input_valid => input_m2s.valid,
        input_last => input_m2s.last,
        input_data => input_m2s.data,

        output_ready => output_ready,
        output_valid => output_m2s.valid,
        output_last => output_m2s.last,
        output_data => output_m2s.data
      );

    output_m2s.user <= (others => '-');

    -- 'done' for the channel, raised where the last beat is accepted by
    -- the *consumer* -- here, past the skid register -- and not where
    -- 'cnn_accel_tensor_mem' sees it accepted, which since this stage was
    -- inserted is one or two beats earlier.
    --
    -- That distinction is invisible on the ifmap channel: 'cmd_proc's
    -- feeder only uses 'src_done' to decide when to issue the *next*
    -- '(row, tile)' request, and every beat still arrives, in order, and
    -- is still consumed. It is fatal on the weight channel. A convolution
    -- with 'space_wgt = LOCAL_TENSOR' fetches its weight, bias and scale
    -- images through 'r1' in three back-to-back requests, and 'cmd_proc'
    -- ends each one on 'side_done' by clearing 'wgt_fill_active_q'. With
    -- the early pulse, the last beat of a region is still sitting in this
    -- register when its request is declared finished, the weight
    -- serializer stops accepting, and that beat is then handed to the
    -- *next* region as its first word -- so every fill after the first is
    -- shifted by a word and the convolution runs on rubbish. It has never
    -- been seen because 'program.py' only ever emitted DDR weights, whose
    -- read DMA raises 'done' against the real consumer; the first tiled
    -- case to ask for resident weights failed on it immediately.
    output_done <= output_m2s.valid and output_m2s.last and output_ready;

    r0_gen : if r = 0 generate
      tm_r0_raw_s2m.ready <= input_ready;
      tm_r0_m2s <= output_m2s;
      tm_r0_done <= output_done;
    end generate;

    r1_gen : if r = 1 generate
      tm_r1_raw_s2m.ready <= input_ready;
      tm_r1_m2s <= output_m2s;
      tm_r1_done <= output_done;
    end generate;
  end generate;

  ------------------------------------------------------------------------
  -- Skid stage on each scratchpad write stream.
  --
  -- Every producer that can write the scratchpad -- the ifmap read DMA's
  -- FIFO, the scratchpad's own read port on a local-to-local move, the
  -- elementwise engine, the conv/pool epilogue -- reaches
  -- 'cnn_accel_tensor_mem's block-RAM write port through 'cnn_accel_cmd_proc's
  -- operand-port binding mux. That was one combinational path from a
  -- producer's output register, across the mux, to a 'DIADI' pin on the
  -- other side of the die: five separate groups of failing endpoints in
  -- the first top-level build, all of them mostly route delay.
  --
  -- 'full_throughput' with both control and data pipelined is a skid
  -- buffer: one beat per cycle sustained, no combinational path from
  -- 'ready' to 'ready' or from 'data' to 'data'. Bit-exact -- a stream
  -- pipeline neither creates, drops nor reorders beats, and the write
  -- request that sizes the transfer is issued before the first data beat
  -- either way, so all that changes is when the beats land.
  ------------------------------------------------------------------------

  tm_write_pipeline_gen : for w in 0 to 1 generate
    signal input_m2s, output_m2s : axi_stream_m2s_t;
    signal input_ready, output_ready : std_ulogic;
  begin
    input_m2s <= tm_w0_m2s when w = 0 else tm_w1_m2s;
    output_ready <= tm_w0_piped_s2m.ready when w = 0 else tm_w1_piped_s2m.ready;

    pipeline_inst : entity common.handshake_pipeline
      generic map (
        data_width => axi_stream_data_sz,
        full_throughput => true,
        pipeline_control_signals => true,
        pipeline_data_signals => true
      )
      port map (
        clk => clk,

        input_ready => input_ready,
        input_valid => input_m2s.valid,
        input_last => input_m2s.last,
        input_data => input_m2s.data,

        output_ready => output_ready,
        output_valid => output_m2s.valid,
        output_last => output_m2s.last,
        output_data => output_m2s.data
      );

    output_m2s.user <= (others => '-');

    w0_gen : if w = 0 generate
      tm_w0_s2m.ready <= input_ready;
      tm_w0_piped_m2s <= output_m2s;
    end generate;

    w1_gen : if w = 1 generate
      tm_w1_s2m.ready <= input_ready;
      tm_w1_piped_m2s <= output_m2s;
    end generate;
  end generate;

  ------------------------------------------------------------------------
  -- Pool-window ready: one lane's, not the AND of all of them.
  --
  -- The lanes are structurally identical and see identical inputs -- the
  -- same 'valid', the same 'first_tile'/'last_tile', the same
  -- configuration -- and a lane's 'ready' is a function of its occupancy
  -- counter alone, never of the activation values it was handed. So
  -- 'pool_lane_ready_vec' is 'g_tile_channels' copies of a single bit,
  -- and taking lane 0 is not a weakening of the AND: it is the same bit.
  --
  -- The AND was previously described here as "a zero-cost statement of
  -- that invariant rather than real arbitration". The invariant is real;
  -- the zero-cost part was not. Routed P&R put the reduction on the
  -- design's worst setup path:
  --
  --   pool_lane_gen[0].pool_inst/buf_count_q  (lane 0's occupancy)
  --     -> lane 0's 'ready'
  --     -> the reduction, which placement had scattered across the lanes
  --     -> 'pool_window_s2m.ready', back at the top level
  --     -> 'pool_window_gen_inst/walk_start' (fanout 67)
  --     -> the reset pin of 'out_col_q', the output-column counter.
  --
  -- Eight logic levels, but 5.0 ns of the 6.4 ns was routing: the AND
  -- made a signal that lives in one lane depend on all the others, so the
  -- chain crossed the die to collect bits it already knew, then crossed
  -- back to steer the walk. Reading one lane deletes those hops and the
  -- LUTs that performed them, and costs nothing -- no register, no
  -- latency, no extra logic.
  --
  -- What is *not* free is the assumption, so it is checked rather than
  -- trusted: 'pool_lane_ready_check' below fails the run the moment two
  -- lanes disagree. That is a strictly better deal than the AND, which
  -- silently absorbed a divergence into a stall and would have turned a
  -- lane bug into a mysterious throughput loss instead of a test failure.
  ------------------------------------------------------------------------

  pool_window_s2m.ready <= pool_lane_ready_vec(0);

  pool_lane_ready_check : process(clk)
  begin
    if rising_edge(clk) then
      for lane in pool_lane_ready_vec'range loop
        assert pool_lane_ready_vec(lane) = pool_lane_ready_vec(0)
          report "cnn_accel_top: pool lane " & natural'image(lane) & " disagrees with " &
            "lane 0 about 'ready'; the pool lanes are required to be in lockstep, and " &
            "'pool_window_s2m.ready' is taken from lane 0 alone"
          severity failure;
      end loop;
    end if;
  end process;


  pool_lane_gen : for lane in 0 to g_tile_channels - 1 generate

    -- This lane's window: the same tap array 'cnn_accel_window_gen'
    -- produces, de-interleaved down to one channel. Declared per lane
    -- (see 'pool_lane_ready_vec') and driven by a static, elaboration-time
    -- index expression, so this is pure wiring, not a multiplexer.
    signal lane_window_m2s : window_m2s_t(data(0 to c_max_taps - 1));
    signal lane_window_s2m : window_s2m_t;

  begin

    lane_window_m2s.valid <= pool_window_m2s.valid;
    lane_window_m2s.last <= pool_window_m2s.last;
    lane_window_m2s.first_tile <= pool_window_m2s.first_tile;
    lane_window_m2s.last_tile <= pool_window_m2s.last_tile;

    lane_slice_gen : for tap in 0 to c_max_taps - 1 generate
      lane_window_m2s.data(tap) <= pool_window_m2s.data(tap * g_tile_channels + lane);
    end generate;

    pool_lane_ready_vec(lane) <= lane_window_s2m.ready;

    pool_inst : entity cnn_accel.cnn_accel_pool
      generic map (
        g_max_kernel_size => g_max_pool_kernel_size,
        g_accum_width => g_accum_width
      )
      port map (
        clk => clk,
        reset => reset_internal,

        cfg_opcode => pool_cfg_opcode,
        cfg_pool_kernel_h => pool_cfg_kernel_h,
        cfg_pool_kernel_w => pool_cfg_kernel_w,

        s_window_m2s => lane_window_m2s,
        s_window_s2m => lane_window_s2m,

        m_max_m2s => pool_lane_max_m2s(lane),
        m_max_s2m => pool_max_s2m,

        m_avgsum_m2s => pool_lane_avgsum_m2s(lane),
        m_avgsum_s2m => pool_avgsum_s2m
      );

  end generate;

  -- POOL_MAX: the lanes' int8 results are already the final output; pack one
  -- activation word and bypass the requantizer entirely, per the pool
  -- requirement document.
  pool_max_pack : process(all)
  begin
    pool_max_m2s <= axi_stream_m2s_init;
    pool_max_m2s.valid <= pool_lane_max_m2s(0).valid;
    pool_max_m2s.last <= pool_lane_max_m2s(0).last;
    pool_max_m2s.data <= (others => '0');
    for lane in 0 to g_tile_channels - 1 loop
      pool_max_m2s.data(8 * lane + 7 downto 8 * lane) <=
        pool_lane_max_m2s(lane).data(7 downto 0);
    end loop;
  end process;

  -- POOL_AVG: the lanes' int32 sums are one 'accum_m2s_t' beat, which is
  -- literally the conv epilogue's input type.
  pool_avgsum_pack : process(all)
  begin
    pool_accum_m2s.valid <= pool_lane_avgsum_m2s(0).valid;
    pool_accum_m2s.last <= pool_lane_avgsum_m2s(0).last;
    for lane in 0 to g_tile_channels - 1 loop
      pool_accum_m2s.data(lane) <=
        signed(pool_lane_avgsum_m2s(lane).data(g_accum_width - 1 downto 0));
    end loop;
  end process;

  pool_avgsum_s2m.ready <= pool_accum_s2m.ready;

  pool_requant_inst : entity cnn_accel.cnn_accel_bias_requant
    generic map (
      g_accum_width => g_accum_width,
      g_pe_rows => g_tile_channels,
      -- No bias table is read on this path, so the address port degenerates to
      -- its minimum legal width.
      g_bias_addr_width => 1,
      g_max_requant_shift => g_max_requant_shift
    )
    port map (
      clk => clk,
      reset => reset_internal,

      -- Division by the pool area is the whole job here: no bias, no ReLU, no
      -- output offset, no clamp and no per-channel table -- 'cmd_proc' has
      -- already turned the area into the scale/shift pair.
      cfg_bias_en => '0',
      cfg_requant_en => '1',
      cfg_relu_en => '0',
      cfg_requant_scale => pool_cfg_requant_scale,
      cfg_requant_shift => pool_cfg_requant_shift,
      cfg_output_offset => (others => '0'),
      cfg_clamp_en => '0',
      cfg_clamp_min => (others => '0'),
      cfg_clamp_max => (others => '0'),
      cfg_per_channel_en => '0',

      bias_rd_addr => open,
      bias_rd_data => (others => '0'),
      scale_rd_data => (others => '0'),

      s_accum_m2s => pool_accum_m2s,
      s_accum_s2m => pool_accum_s2m,

      m_out_m2s => pool_avg_m2s,
      m_out_s2m => pool_avg_s2m
    );

  -- Final POOL_MAX / POOL_AVG output select. 'cnn_accel_pool' already makes
  -- the two paths structurally mutually exclusive (one shared output register,
  -- tagged), so this is a select, not an arbiter.
  pool_is_avg <= '1' when pool_cfg_opcode = c_opcode_pool_avg else '0';

  pool_out_m2s <= pool_avg_m2s when pool_is_avg = '1' else pool_max_m2s;
  pool_avg_s2m.ready <= pool_out_s2m.ready and pool_is_avg;
  pool_max_s2m.ready <= pool_out_s2m.ready and not pool_is_avg;

  -- Same completion rule as 'cnn_accel_conv_core': the pass is over when the
  -- engine's last beat has been accepted by its sink.
  pool_done <= pool_out_m2s.valid and pool_out_m2s.last and pool_out_s2m.ready;

  ------------------------------------------------------------------------
  -- Elementwise engine (ADD / UPSAMPLE / COPY / ACT). Unlike the other two it
  -- is its own address generator: it issues the DMA requests and 'cmd_proc'
  -- only resolves their space tags onto real ports, which is why the request
  -- records flow from this block towards 'cmd_proc'.
  ------------------------------------------------------------------------

  elementwise_inst : entity cnn_accel.cnn_accel_elementwise
    generic map (
      g_axi_data_width => g_axi_data_width,
      -- 'g_max_requant_shift' and 'g_max_xfer_bytes' keep their block-level
      -- defaults: the first is the elementwise datapath's own documented
      -- bound (narrower than the conv epilogue's) and the second is an
      -- internal counter bound, not an address-space bound -- the address
      -- space is policed by 'cmd_proc's 'g_ddr_limit'/'g_tensor_bytes'.
      g_max_xfer_bytes => g_ddr_limit
    )
    port map (
      clk => clk,
      reset => reset_internal,

      start => ew_start,
      opcode => ew_opcode,
      src0_addr => ew_src0_addr,
      src1_addr => ew_src1_addr,
      dst_addr => ew_dst_addr,
      lut_addr => ew_lut_addr,
      xfer_bytes => ew_xfer_bytes,
      in_width => ew_in_width,
      in_height => ew_in_height,
      in_channels => ew_in_channels,
      requant_scale => ew_requant_scale,
      requant_shift => ew_requant_shift,

      done => ew_done,
      error => ew_error,
      error_code => ew_error_code,

      src0_req_m2s => ew_src0_req_m2s,
      src0_req_s2m => ew_src0_req_s2m,
      s_src0_stream_m2s => ew_src0_stream_m2s,
      s_src0_stream_s2m => ew_src0_stream_s2m,

      src1_req_m2s => ew_src1_req_m2s,
      src1_req_s2m => ew_src1_req_s2m,
      s_src1_stream_m2s => ew_src1_stream_m2s,
      s_src1_stream_s2m => ew_src1_stream_s2m,

      dst_req_m2s => ew_dst_req_m2s,
      dst_req_s2m => ew_dst_req_s2m,
      m_dst_stream_m2s => ew_dst_stream_m2s,
      m_dst_stream_s2m => ew_dst_stream_s2m,

      lut_req_m2s => ew_lut_req_m2s,
      lut_req_s2m => ew_lut_req_s2m,
      s_lut_stream_m2s => ew_lut_stream_m2s,
      s_lut_stream_s2m => ew_lut_stream_s2m
    );

end architecture a;
