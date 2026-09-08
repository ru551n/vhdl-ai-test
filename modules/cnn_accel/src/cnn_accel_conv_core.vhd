library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Convolution datapath composition ("M6b"): direct entity instantiation of
-- cnn_accel_window_gen -> cnn_accel_pe_array -> cnn_accel_bias_requant, with
-- a cnn_accel_weight_buffer instance feeding pe_array's weight/bias read
-- ports. See modules/cnn_accel/doc/cnn_accel_tiled_dataflow_proposal.md
-- (the ratified dataflow this composition implements, no redesign here) and
-- each submodule's own entity-level comment for its individual contract.
-- Composition only: this entity adds no new datapath logic of its own
-- beyond wiring and the elaboration-time cross-module asserts below.
--
-- Scope (deliberately excluded, later milestones):
--   * No AXI4/AXI4-Lite/register interface, no DMA, no instruction decode,
--     no layer sequencing -- configuration is plain 'cfg_*' ports and a
--     'start' pulse, exactly as cnn_accel_window_gen already takes them.
--   * cnn_accel_pool is not part of this entity.
--   * Output-channel tiling (proposal section 5, D6) is a LAYER-LEVEL
--     concern: a future cnn_accel_layer_ctrl re-streams the whole ifmap
--     through this entity once per output-channel tile, each pass filling
--     cnn_accel_weight_buffer with that tile's own weight/bias slice of
--     'weights_packed.txt'/'pack_bias_for_hw' (cnn_accel_model.py) via
--     'fill_start'/'fill_is_bias' before pulsing 'start' again. This
--     entity does not build that loop, and (inherited unavoidably from
--     cnn_accel_bias_requant's own v1 design, see its entity-level comment)
--     cannot itself be driven with more than one output-channel tile's
--     worth of output channels in a single 'start': cnn_accel_bias_requant
--     always reads bias row 0 (assumes out_channels <= g_pe_rows for the
--     whole layer), and cnn_accel_pe_array's own 'weight_rd_addr' restarts
--     at 0 on every pixel's 'first_tile' beat, never offset by an
--     output-channel-tile index -- neither submodule has a port for one, so
--     no such port is invented here either (adding one to only this
--     composition entity would be a half-implemented feature: the address
--     would have nowhere correct to go on the bias side). A future
--     layer_ctrl re-streaming loop that needs one drives it entirely by
--     re-filling cnn_accel_weight_buffer with the next tile's weight/bias
--     slice between passes, not by an address offset port on this entity.
--
-- Handshake composition (shared/Axi4.md rules 20-22/26): every internal
-- link connects two submodules whose handshake was already designed, by the
-- ratified proposal, for exactly this connection -- no skid/handshake
-- register (e.g. hdl-modules' common.handshake_pipeline) is needed anywhere
-- in this composition:
--   * s_stream (activation ingest) and m_out (requantized output) and
--     s_weight (weight/bias fill) are passed straight through to/from
--     cnn_accel_window_gen/cnn_accel_bias_requant/cnn_accel_weight_buffer's
--     own AXI4-Stream ports -- each entity's own 'ready' is already a pure
--     function of its own registered state (window_gen: 'active_q' and the
--     pending-window register; weight_buffer: the fill row pointer;
--     bias_requant: its one-entry output register), so exposing them
--     unmodified adds no new combinational-loop risk.
--   * window_gen's 'm_window' -> pe_array's 's_window': window_gen's
--     'window_valid' is a pure function of registered row/column counters
--     (cnn_accel_window_gen.vhd:459), and pe_array's 's_window_s2m.ready' is
--     a pure function of its own 'state_q' (cnn_accel_pe_array.vhd:281) --
--     exactly the link doc/cnn_accel_tiled_dataflow_proposal.md section 7
--     describes as "unchanged in shape from today", designed for direct
--     connection.
--   * pe_array's 'm_accum' -> bias_requant's 's_accum': both sides already
--     implement a one-entry output/flow-through register with 'ready' a
--     pure function of that register's own state plus the downstream
--     'ready' input (pe_array's 'can_commit_v', cnn_accel_pe_array.vhd:319;
--     bias_requant's 'accept_input', cnn_accel_bias_requant.vhd:288) -- the
--     same "flow-through, no bubble" idiom on both ends of one link,
--     designed to compose directly (proposal section 7: "unchanged from the
--     pre-tiling proposal").
--   * weight_rd_addr/weight_rd_data and bias_rd_addr/bias_rd_data/
--     scale_rd_data (ISA v1.2) are plain
--     synchronous read ports (no handshake at all, 1 cycle registered
--     latency, always-ready -- cnn_accel_weight_buffer's own contract),
--     wired straight across.
--
-- Cross-module generic contracts: every generic that must be IDENTICAL
-- across two submodule instances (g_max_kernel_size, g_tile_channels,
-- g_pe_rows, g_pe_cols, g_accum_width, g_weight_buffer_depth,
-- g_bias_buffer_depth) is a single generic on THIS entity, fed unmodified
-- to every instance that needs it --
-- so those contracts are structurally enforced (a single source of truth),
-- not merely asserted. 'g_bias_addr_width' (cnn_accel_bias_requant) is not
-- a separate generic here at all: it is derived locally from
-- 'g_weight_buffer_depth' with the exact same formula
-- cnn_accel_weight_buffer/cnn_accel_pe_array use for their own
-- 'weight_rd_addr'/'bias_rd_addr' widths, so the three ports this entity
-- wires together (pe_array's 'weight_rd_addr', weight_buffer's
-- 'weight_rd_addr'/'bias_rd_addr', bias_requant's 'bias_rd_addr') are
-- guaranteed the same width by construction, not by a runtime assert.
--
-- The one generic RELATIONSHIP that is not structurally forced (different
-- generics, only conventionally recommended equal) gets an explicit
-- elaboration-time assert below: 'g_tile_channels' vs. 'g_pe_cols'
-- (doc/cnn_accel_tiled_dataflow_proposal.md section 3/8 risk 2).
entity cnn_accel_conv_core is
  generic (
    -- Output-channel parallelism (cnn_accel_pe_array/cnn_accel_
    -- weight_buffer/cnn_accel_bias_requant's own 'g_pe_rows').
    g_pe_rows : positive;
    -- Input-channel/MAC parallelism per cycle (cnn_accel_pe_array/
    -- cnn_accel_weight_buffer's own 'g_pe_cols').
    g_pe_cols : positive;
    -- Accumulator width (int32 default).
    g_accum_width : positive := 32;
    -- Upper bound on 'cfg_kernel_h'/'cfg_kernel_w' (cnn_accel_window_gen/
    -- cnn_accel_pe_array's own 'g_max_kernel_size').
    g_max_kernel_size : positive;
    -- Input channels processed in parallel per beat/tile ("Ct")
    -- (cnn_accel_window_gen/cnn_accel_pe_array's own 'g_tile_channels').
    -- Recommended equal to 'g_pe_cols' -- see the elaboration-time assert
    -- below.
    g_tile_channels : positive;
    -- Upper bound on 'cfg_in_width * ceil(cfg_in_channels/g_tile_channels)'
    -- (cnn_accel_window_gen's own 'g_max_row_tile_words').
    g_max_row_tile_words : positive;
    -- Rows per cnn_accel_weight_buffer weight region (cnn_accel_weight_
    -- buffer/cnn_accel_pe_array's own 'g_weight_buffer_depth'). Must cover
    -- the whole layer's rows ('T * groups_per_tile'), per
    -- doc/cnn_accel_tiled_dataflow_proposal.md section 4 -- checked at
    -- runtime by cnn_accel_pe_array's own 'severity failure' assert.
    g_weight_buffer_depth : positive;
    -- Rows in cnn_accel_weight_buffer's (separate, much shallower) bias
    -- region -- cnn_accel_weight_buffer's own 'g_bias_buffer_depth',
    -- forwarded unmodified so 'bias_rd_addr''s width here matches that
    -- entity's actual bias-region address width by construction.
    g_bias_buffer_depth : positive := 8;
    -- Upper bound on the runtime-variable 'cfg_requant_shift'
    -- (cnn_accel_bias_requant's own 'g_max_requant_shift').
    g_max_requant_shift : natural := 31
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level),
    -- fanned out unmodified to every submodule instance.
    reset : in std_ulogic := '0';
    --# {{}}
    -- Kernel/stride/padding/frame-size configuration, latched at 'start' by
    -- cnn_accel_window_gen; 'cfg_kernel_h'/'cfg_kernel_w' are also sampled
    -- directly (combinationally, at 's_window' accept time) by
    -- cnn_accel_pe_array -- see that entity's own port comment.
    cfg_kernel_h : in std_ulogic_vector(7 downto 0);
    cfg_kernel_w : in std_ulogic_vector(7 downto 0);
    cfg_stride_h : in std_ulogic_vector(7 downto 0);
    cfg_stride_w : in std_ulogic_vector(7 downto 0);
    cfg_pad_top : in std_ulogic_vector(7 downto 0);
    cfg_pad_bottom : in std_ulogic_vector(7 downto 0);
    cfg_pad_left : in std_ulogic_vector(7 downto 0);
    cfg_pad_right : in std_ulogic_vector(7 downto 0);
    cfg_in_width : in std_ulogic_vector(15 downto 0);
    cfg_in_height : in std_ulogic_vector(15 downto 0);
    cfg_in_channels : in std_ulogic_vector(15 downto 0);
    --# {{}}
    -- Output-quantization configuration for cnn_accel_bias_requant, sampled
    -- combinationally per accepted beat (not latched at 'start' -- see that
    -- entity's own header comment).
    cfg_bias_en : in std_ulogic;
    cfg_requant_en : in std_ulogic;
    cfg_relu_en : in std_ulogic;
    cfg_requant_scale : in std_ulogic_vector(31 downto 0);
    cfg_requant_shift : in std_ulogic_vector(7 downto 0);
    -- ISA v1.1 (H1) epilogue fields (instruction word W13 + FLAG_CLAMP_EN);
    -- all-zero reproduces the v1.0 epilogue exactly.
    cfg_output_offset : in std_ulogic_vector(15 downto 0) := (others => '0');
    cfg_clamp_en : in std_ulogic := '0';
    cfg_clamp_min : in std_ulogic_vector(7 downto 0) := (others => '0');
    cfg_clamp_max : in std_ulogic_vector(7 downto 0) := (others => '0');
    -- ISA v1.2 (H2) FLAG_PER_CHANNEL_EN: cnn_accel_bias_requant takes each
    -- lane's (multiplier, shift) from cnn_accel_weight_buffer's scale
    -- region (filled through 's_weight' with 'fill_is_scale') instead of
    -- 'cfg_requant_scale'/'cfg_requant_shift'. '0' is the pre-H2 datapath.
    cfg_per_channel_en : in std_ulogic := '0';
    --# {{}}
    -- Pulse: latches the 'cfg_*' ports above and resets cnn_accel_
    -- window_gen's row/column counters and line-buffer pointers for a new
    -- frame (unmodified pass-through of that entity's own 'start' port).
    start : in std_ulogic;
    -- Pulse: the final requantized output beat of the frame ('m_out_m2s.
    -- last') has been accepted ('m_out_s2m.ready' the same cycle). NOT the
    -- same signal as cnn_accel_window_gen's own internal 'done' (which
    -- only means the final INPUT window has been accepted by
    -- cnn_accel_pe_array -- pe_array/bias_requant's own pipeline latency
    -- and one-entry output registers mean real output can still be
    -- in-flight well after that); window_gen's 'done' is not a port of
    -- this entity for exactly that reason -- it would be misleading at
    -- this boundary.
    done : out std_ulogic;
    --# {{}}
    -- Raster-order int8 input activations, unmodified pass-through of
    -- cnn_accel_window_gen's own 's_stream' port -- see that entity's own
    -- port comment for the exact one-beat-per-'(col,tile)'-cell contract.
    s_stream_m2s : in axi_stream_m2s_t;
    s_stream_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- Weight/bias preload fill port, unmodified pass-through of
    -- cnn_accel_weight_buffer's own 's_stream'/'fill_start'/
    -- 'fill_is_bias' ports -- see that entity's own entity-level comment.
    -- Exposed so a testbench (this milestone) or a future
    -- cnn_accel_axi_read_dma instance (a later milestone) can preload
    -- weights/biases; no fill sequencer is built here.
    fill_start : in std_ulogic := '0';
    fill_is_bias : in std_ulogic;
    -- ISA v1.2 (H2): routes fill beats to the per-channel scale region
    -- (one 8-byte table entry, 'data(39 downto 0)', per beat) -- see
    -- cnn_accel_weight_buffer's own port comment.
    fill_is_scale : in std_ulogic := '0';
    s_weight_m2s : in axi_stream_m2s_t;
    s_weight_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- Requantized int8 output activations, unmodified pass-through of
    -- cnn_accel_bias_requant's own 'm_out' port.
    m_out_m2s : out axi_stream_m2s_t;
    m_out_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_conv_core;

architecture a of cnn_accel_conv_core is

  ------------------------------------------------------------------------
  -- Local sizing constants -- see the entity-level comment on why these
  -- are not separate generics.
  ------------------------------------------------------------------------

  constant c_window_len : positive := window_data_length(g_max_kernel_size, g_tile_channels);
  constant c_addr_width : positive := num_bits_needed(g_weight_buffer_depth - 1);
  constant c_bias_addr_width : positive := num_bits_needed(g_bias_buffer_depth - 1);
  constant c_weight_lanes : positive := g_pe_rows * g_pe_cols;

  ------------------------------------------------------------------------
  -- Inter-submodule links.
  ------------------------------------------------------------------------

  signal window_m2s : window_m2s_t(data(0 to c_window_len - 1));
  signal window_s2m : window_s2m_t;

  signal weight_rd_addr : std_ulogic_vector(c_addr_width - 1 downto 0);
  signal weight_rd_data : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0);
  signal bias_rd_addr : std_ulogic_vector(c_bias_addr_width - 1 downto 0);
  signal bias_rd_data : std_ulogic_vector(g_accum_width * g_pe_rows - 1 downto 0);
  signal scale_rd_data : std_ulogic_vector(c_scale_entry_width * g_pe_rows - 1 downto 0);

  signal accum_m2s : accum_m2s_t(data(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0));
  signal accum_s2m : accum_s2m_t;

begin

  ------------------------------------------------------------------------
  -- Cross-module generic contract that is NOT structurally forced (see
  -- entity-level comment): a mismatch is not a functional bug (pe_array's
  -- own group masking already handles a non-dividing 'kh*kw*g_tile_channels
  -- / g_pe_cols', proven by tb_cnn_accel_pe_array.vhd's own deliberately
  -- mismatched g_tile_channels=6/g_pe_cols=4 generics), only a lost-cycle
  -- efficiency concern -- hence 'severity warning', not 'failure'.
  ------------------------------------------------------------------------

  assert g_tile_channels = g_pe_cols
    report "cnn_accel_conv_core: g_tile_channels (" & positive'image(g_tile_channels) &
      ") != g_pe_cols (" & positive'image(g_pe_cols) &
      ") -- doc/cnn_accel_tiled_dataflow_proposal.md section 3/8 risk 2 recommends " &
      "equality for an exact (no-remainder) groups-per-tile fit; still functionally " &
      "correct otherwise (cnn_accel_pe_array masks the non-dividing remainder group), " &
      "just leaves PE cycles idle in that group"
    severity warning;

  ------------------------------------------------------------------------
  -- cnn_accel_window_gen: raster-order int8 activations -> tiled K_h x K_w
  -- x Ct windows.
  ------------------------------------------------------------------------

  window_gen_inst : entity cnn_accel.cnn_accel_window_gen
    generic map (
      g_max_kernel_size => g_max_kernel_size,
      g_max_row_tile_words => g_max_row_tile_words,
      g_tile_channels => g_tile_channels
    )
    port map (
      clk => clk,
      reset => reset,

      cfg_kernel_h => cfg_kernel_h,
      cfg_kernel_w => cfg_kernel_w,
      cfg_stride_h => cfg_stride_h,
      cfg_stride_w => cfg_stride_w,
      cfg_pad_top => cfg_pad_top,
      cfg_pad_bottom => cfg_pad_bottom,
      cfg_pad_left => cfg_pad_left,
      cfg_pad_right => cfg_pad_right,
      cfg_in_width => cfg_in_width,
      cfg_in_height => cfg_in_height,
      cfg_in_channels => cfg_in_channels,

      start => start,
      -- Not this entity's 'done' -- see the port comment above.
      done => open,

      s_stream_m2s => s_stream_m2s,
      s_stream_s2m => s_stream_s2m,

      m_window_m2s => window_m2s,
      m_window_s2m => window_s2m
    );

  ------------------------------------------------------------------------
  -- cnn_accel_pe_array: tiled windows -> per-pixel int32 partial-sum
  -- accumulators, one input-channel-tile-partial-sum-carry pass per pixel.
  ------------------------------------------------------------------------

  pe_array_inst : entity cnn_accel.cnn_accel_pe_array
    generic map (
      g_pe_rows => g_pe_rows,
      g_pe_cols => g_pe_cols,
      g_accum_width => g_accum_width,
      g_max_kernel_size => g_max_kernel_size,
      g_tile_channels => g_tile_channels,
      g_weight_buffer_depth => g_weight_buffer_depth
    )
    port map (
      clk => clk,
      reset => reset,

      cfg_kernel_h => cfg_kernel_h,
      cfg_kernel_w => cfg_kernel_w,

      s_window_m2s => window_m2s,
      s_window_s2m => window_s2m,

      weight_rd_addr => weight_rd_addr,
      weight_rd_data => weight_rd_data,

      m_accum_m2s => accum_m2s,
      m_accum_s2m => accum_s2m
    );

  ------------------------------------------------------------------------
  -- cnn_accel_weight_buffer: single-buffered on-chip weight/bias cache
  -- (with a shallow prefetch FIFO on the fill stream). Fill side exposed
  -- on this entity's own ports; read side feeds pe_array's/bias_requant's
  -- read-only ports directly (no handshake).
  ------------------------------------------------------------------------

  weight_buffer_inst : entity cnn_accel.cnn_accel_weight_buffer
    generic map (
      g_weight_buffer_depth => g_weight_buffer_depth,
      g_bias_buffer_depth => g_bias_buffer_depth,
      g_pe_rows => g_pe_rows,
      g_pe_cols => g_pe_cols,
      g_accum_width => g_accum_width
    )
    port map (
      clk => clk,
      reset => reset,

      s_stream_m2s => s_weight_m2s,
      s_stream_s2m => s_weight_s2m,

      fill_start => fill_start,
      fill_is_bias => fill_is_bias,
      fill_is_scale => fill_is_scale,

      weight_rd_addr => weight_rd_addr,
      weight_rd_data => weight_rd_data,

      bias_rd_addr => bias_rd_addr,
      bias_rd_data => bias_rd_data,
      scale_rd_data => scale_rd_data
    );

  ------------------------------------------------------------------------
  -- cnn_accel_bias_requant: per-pixel int32 accumulators -> bias/requant/
  -- offset/clamp (ReLU+saturate in v1.0 terms) -> requantized int8 output
  -- stream.
  ------------------------------------------------------------------------

  bias_requant_inst : entity cnn_accel.cnn_accel_bias_requant
    generic map (
      g_accum_width => g_accum_width,
      g_pe_rows => g_pe_rows,
      g_bias_addr_width => c_bias_addr_width,
      g_max_requant_shift => g_max_requant_shift
    )
    port map (
      clk => clk,
      reset => reset,

      cfg_bias_en => cfg_bias_en,
      cfg_requant_en => cfg_requant_en,
      cfg_relu_en => cfg_relu_en,
      cfg_requant_scale => cfg_requant_scale,
      cfg_requant_shift => cfg_requant_shift,
      cfg_output_offset => cfg_output_offset,
      cfg_clamp_en => cfg_clamp_en,
      cfg_clamp_min => cfg_clamp_min,
      cfg_clamp_max => cfg_clamp_max,
      cfg_per_channel_en => cfg_per_channel_en,

      bias_rd_addr => bias_rd_addr,
      bias_rd_data => bias_rd_data,
      scale_rd_data => scale_rd_data,

      s_accum_m2s => accum_m2s,
      s_accum_s2m => accum_s2m,

      m_out_m2s => m_out_m2s,
      m_out_s2m => m_out_s2m
    );

  ------------------------------------------------------------------------
  -- Whole-frame completion pulse -- see the 'done' port comment above.
  -- Combinational function of the output-side transfer condition (rule 1,
  -- shared/Axi4.md), not a new registered signal; introduces no
  -- combinational loop since it only fans out (nothing feeds back from it).
  ------------------------------------------------------------------------

  done <= m_out_m2s.valid and m_out_m2s.last and m_out_s2m.ready;

end architecture a;
