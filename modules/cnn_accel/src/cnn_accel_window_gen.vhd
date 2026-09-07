library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Configurable K_h x K_w / stride / zero-padding / input-channel-tiled
-- sliding-window generator. See modules/cnn_accel/doc/cnn_accel_window_gen_req.md,
-- doc/cnn_accel_window_gen_proposal.md (pre-tiling design) and
-- doc/cnn_accel_tiled_dataflow_proposal.md sections 1/2/9 (the channel-
-- tiling retrofit implemented here).
--
-- Architecture: 'g_max_kernel_size' full-row banks (BRAM-inference intent,
-- each 'g_max_row_tile_words' *channel-tile cells* wide -- 'in_width *
-- ceil(in_channels/g_tile_channels)' cells, not one cell per column, see
-- below), ping-ponged by physical input row number modulo
-- 'g_max_kernel_size' -- literally the requirement doc's own "buffers
-- K_h - 1 full rows ... plus the current row ... ping-pong across K_h row
-- banks" description, not a FIFO/shift-register pipeline (see
-- cnn_accel_window_gen_proposal.md section 4 for why a FIFO-based design,
-- closer to the classic fixed-3x3 line-buffer technique, is insufficient
-- here: a fixed 3x3 window has no configurable stride/padding, so it
-- never needs to *replay* an already-fully-written row/column for more
-- than one output position; this module's padding can make a
-- bottom/right output row or column position's real (unpadded)
-- row/column identical to an earlier one's, which a pop-once FIFO or a
-- shallow shift register can no longer supply once its head has moved
-- on, but a full-row bank -- read (never popped) at whatever address a
-- given tap needs -- still can).
--
-- Channel tiling (cnn_accel_tiled_dataflow_proposal.md sections 1/2):
-- 'cfg_in_channels' need not equal 'g_tile_channels'. Each output pixel
-- emits 'T = ceil(cfg_in_channels / g_tile_channels)' consecutive
-- 'm_window' beats, one per input-channel tile, with 'first_tile'/
-- 'last_tile' sidebands marking the first/last tile of that pixel (both
-- '1' when 'T = 1'). The stream-level 'last' flag still means "last
-- output pixel of the whole feature map" and is asserted only on the
-- final beat ('last_tile' = '1') of the final pixel -- never on every
-- 'last_tile'. Symmetrically, one accepted 's_stream' beat now writes one
-- '(col, tile)' cell (not one full pixel): 'T' beats/pixel on ingest too.
-- When 'cfg_in_channels' is not a multiple of 'g_tile_channels', the
-- final tile's unused channel lanes are driven to '0' at write time
-- (D11) -- deterministically, not left as whatever garbage
-- 's_stream_m2s.data' happens to carry above the valid byte range.
--
-- M7 (doc/cnn_accel_window_gen_bram_proposal.md, ratified Option 3a):
-- the row banks are read through a *registered* single-read-port
-- 'memory_block' idiom (matches cnn_accel_weight_buffer.vhd), one read
-- per bank per cycle. Since a window needs up to 'g_max_kernel_size'
-- columns per row, the read side walks a registered 'kc' column counter
-- over 'cfg_kernel_w' cycles, reading all 'g_max_kernel_size' banks in
-- parallel each cycle and packing the results into a tap-assembly
-- register; the complete window is then presented as one 'm_window_m2s'
-- beat, same as before. This trades throughput (an extra ~'kw + 1' to
-- 'kw + 2' cycles/window, see the proposal doc's DP3) for block-RAM
-- inference (the combinational random-access read this replaces forced
-- distributed RAM -- see the proposal doc section 1).
entity cnn_accel_window_gen is
  generic (
    -- Upper bound on 'cfg_kernel_h'/'cfg_kernel_w'; sizes the row-bank
    -- count (below) and the window's tap grid. Contract:
    -- 'g_max_kernel_size >= 2' (asserted below).
    g_max_kernel_size : positive;
    -- Upper bound on 'cfg_in_width * ceil(cfg_in_channels / g_tile_channels)'
    -- ("row-tile-word count"); sizes each row bank's depth (BRAM-inference
    -- intent). Bounding the *product* (rather than sizing width and
    -- channel-count independently) is deliberate -- see
    -- cnn_accel_tiled_dataflow_proposal.md section 1. Checked by a
    -- 'severity failure' assert at 'start' (runtime values, not a true
    -- generic-only elaboration bound).
    g_max_row_tile_words : positive;
    -- Input channels processed in parallel per beat/tile ('Ct'). Contract:
    -- 'g_tile_channels * 8 <= axi_stream_data_sz' (asserted below), since
    -- one tile's channels must fit in 's_stream_m2s.data''s low bytes.
    -- 'cfg_in_channels' need not be a multiple of this -- see the
    -- entity-level comment on channel tiling / D11 zero-padding.
    g_tile_channels : positive
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';
    --# {{}}
    -- Kernel/stride/padding/frame-size configuration, latched at 'start'.
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
    -- Pulse, from cnn_accel_layer_ctrl: latches the 'cfg_*' ports above and
    -- resets row/column counters and line-buffer pointers for a new frame.
    start : in std_ulogic;
    -- Pulse: the final tile beat of the final window of the frame has
    -- been accepted (m_window_s2m.ready = '1' the same cycle).
    done : out std_ulogic;
    --# {{}}
    -- Raster-order int8 input pixels, from the ifmap
    -- cnn_accel_axi_read_dma. One accepted beat writes one channel-tile
    -- of one column; 'data' low '8 * g_tile_channels' bits hold that
    -- tile's channels (remaining high bits, if any, are ignored) -- see
    -- the entity-level comment on channel tiling.
    s_stream_m2s : in axi_stream_m2s_t;
    s_stream_s2m : out axi_stream_s2m_t;
    --# {{}}
    -- One K_h x K_w x g_tile_channels window per beat, 'T' consecutive
    -- beats per output pixel (one per input-channel tile). 'data' low
    -- 'cfg_kernel_h * cfg_kernel_w * g_tile_channels * 8' bits hold the
    -- window: tap 'i' (row-major, 'i = row * cfg_kernel_w + col'), channel
    -- 'c' within the tile, at bits '8*(i*g_tile_channels + c) + 7 downto
    -- 8*(i*g_tile_channels + c)'; remaining high bits are '0'. Partial
    -- final-tile lanes ('c' beyond the valid channel count) read as '0'
    -- (D11), not garbage. 'first_tile'/'last_tile' mark the first/last
    -- tile of the current output pixel (both '1' when 'T = 1'). 'last' is
    -- '1' only for the final tile beat of the final window of the frame.
    m_window_m2s : out window_m2s_t(data(0 to window_data_length(g_max_kernel_size, g_tile_channels) - 1));
    m_window_s2m : in window_s2m_t
  );
end entity cnn_accel_window_gen;

architecture a of cnn_accel_window_gen is

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  constant c_lane_width : positive := 8 * g_tile_channels;
  constant c_window_data_length : positive := window_data_length(g_max_kernel_size, g_tile_channels);

  -- S7 timing rework (see the read-address comment block below): every
  -- row-bank cell address is now maintained as an increment-only
  -- accumulator instead of being recomputed by a runtime multiply each
  -- cycle, so read/write addresses are carried *modulo*
  -- '2**c_addr_width' rather than exactly. That is sufficient because
  -- every address the design ever actually *uses* is an in-frame cell,
  -- i.e. '< cfg_in_width * n_tiles <= g_max_row_tile_words <=
  -- 2**c_addr_width' (the bound is asserted at 'start'), so the low
  -- 'c_addr_width' bits of a modular accumulator reproduce it exactly;
  -- out-of-frame accumulator values (the left/right zero-padding
  -- excursions a column walk passes through, where the true value is
  -- negative or beyond the row) are never read out -- 'in_frame_now'
  -- discards them, exactly as the predecessor's 'rd_addr(b) <= 0' branch
  -- did.
  constant c_addr_width : positive := num_bits_needed(g_max_row_tile_words - 1);
  subtype word_t is unsigned(c_addr_width - 1 downto 0);

  -- Signed row/column coordinate. 'cfg_in_width'/'cfg_in_height' are
  -- 16-bit and every kernel/stride/padding value 8-bit, so 18 bits
  -- signed covers every 'row_top'/'col_left'/'input_row'/'input_col'
  -- intermediate below -- including the negative values top/left padding
  -- produces -- without any of them becoming an unbounded 32-bit
  -- 'integer' in the netlist.
  subtype coord_t is integer range -131072 to 131071;

  ------------------------------------------------------------------------
  -- Row banks: 'g_max_kernel_size' full-row buffers, each
  -- 'g_max_row_tile_words' channel-tile cells wide -- cell 'col * n_tiles
  -- + tile' holds column 'col''s tile 'tile' (one channel-tile of int8
  -- activations, 'c_lane_width' bits), not one cell per column -- see
  -- the entity-level comment on channel tiling and
  -- cnn_accel_tiled_dataflow_proposal.md section 1. Physical input row
  -- number 'r' always lives in bank 'r mod g_max_kernel_size'; since
  -- 'cfg_kernel_h <= g_max_kernel_size' (asserted at 'start'), at most
  -- 'g_max_kernel_size' distinct physical rows (the current row plus up
  -- to 'g_max_kernel_size - 1' previous ones) are ever simultaneously
  -- needed by any pending/future output row, so this many banks never
  -- forces an unread row to be evicted -- see the entity-level comment
  -- above and cnn_accel_window_gen_proposal.md section 4.
  --
  -- M7: each bank is the 'memory_block' idiom already used by
  -- cnn_accel_weight_buffer.vhd (:108-116, :245-251) -- one array signal,
  -- one decoded single write per cycle, and a *registered* single read
  -- per cycle, no reset on the array itself.
  --
  -- Correction (post-M7 measurement): a *single* 'row_banks' signal
  -- indexed by a runtime bank number, even when every write/read site
  -- only ever touches one *statically*-numbered element per unrolled
  -- loop iteration, is still one combined multi-dimensional object to
  -- GHDL/Yosys once more than one of those per-iteration accesses is
  -- live simultaneously (this module's read side needs all
  -- 'g_max_kernel_size' banks' data every cycle, in parallel, to feed
  -- the tap-assembly register) -- 'memory_collect' never turned it into
  -- '$mem' cells at all (0 found), and the whole array fell back to
  -- plain flip-flops + mux trees, i.e. exactly the distributed-RAM
  -- blowup this rework was meant to fix. Confirmed with a standalone
  -- 'ghdl ...; proc; memory_collect; stat' probe on this entity: 0
  -- '$mem_v2' cells either way.
  --
  -- Fix: 'g_max_kernel_size' *physically separate* signals, one per
  -- 'gen_banks' generate branch below -- each branch's 'bank_mem' is a
  -- distinct elaborated object (not a slice of a shared array), so each
  -- is its own trivial one-write-port/one-read-port memory candidate,
  -- structurally identical to cnn_accel_weight_buffer.vhd's single
  -- 'weight_mem'/'bias_mem' pair, just replicated by generate instead of
  -- selected by a runtime index. The write enable/address/data feeding
  -- every branch are computed once, combinationally, by 'wr_decode'
  -- below (see its comment for why that logic moved out of 'control').
  ------------------------------------------------------------------------

  type row_bank_t is array (0 to g_max_row_tile_words - 1) of
    std_ulogic_vector(c_lane_width - 1 downto 0);

  -- Registered read output, one word per bank, one cycle after 'rd_addr'
  -- is presented -- the BRAM-inference-critical registered read port
  -- (proposal doc section 2 rule 1). No reset: read-data content has no
  -- completeness contract of its own, same as cnn_accel_weight_buffer.vhd.
  type bank_word_arr_t is array (0 to g_max_kernel_size - 1) of
    std_ulogic_vector(c_lane_width - 1 downto 0);
  signal bank_rd_data : bank_word_arr_t;

  -- S7: ONE shared read address register, not one per bank, driving every
  -- bank's address port directly.
  --
  -- Why (measured, Vivado xc7a200tfbg484-2, 500 MHz constraint): the
  -- predecessor's combinational per-bank 'rd_addr(b)' was the design's
  -- critical path at 20.419 ns / 34 logic levels / 21 CARRY4 -- 47.66 MHz
  -- for the whole 'cnn_accel_conv_core' -- from a DSP48E1's CLK->P
  -- (3.375 ns, MREG but no PREG, i.e. straight through the post-multiply
  -- adder) all the way onto 'gen_banks[2].bank_mem''s ADDRARDADDR pins,
  -- because one cycle had to evaluate 'out_col_q * stride_w_q', then
  -- 'input_col * n_tiles_q', then the per-bank in-frame mux, with nothing
  -- registered in between. Two independent wastes fed that: 'rd_addr(b)'
  -- was the *same value for every bank* (only the 'in_frame_now(b)'
  -- enable differs), and the value itself is an affine function of 'kc_q'
  -- and so wants to be an accumulator, not a multiplier (see
  -- 'rd_word_q').
  signal rd_addr_q : integer range 0 to g_max_row_tile_words - 1 := 0;

  -- Write-side decode. 'wr_bank_q'/'wr_addr_q' are S7 *counters*
  -- (advanced by 'control' on each accepted beat), replacing the former
  -- combinational 'cur_row_q mod g_max_kernel_size' (a runtime modulo of
  -- a 16-bit value by a non-power-of-two) and 'cur_col_q * n_tiles_q +
  -- wr_tile_q' (a second runtime multiply, also landing directly on a
  -- BRAM address port). Neither needs any arithmetic: consecutive
  -- accepted beats write consecutive cells of one row, restarting at
  -- cell 0 at every input-row boundary, where the bank number advances
  -- one step around a mod-'g_max_kernel_size' ring. 'wr_data_c' stays
  -- combinational -- it is a byte-lane mask on 's_stream_m2s.data' (D11)
  -- feeding DI, not ADDR.
  signal wr_bank_q : natural range 0 to g_max_kernel_size - 1 := 0;
  signal wr_addr_q : integer range 0 to g_max_row_tile_words - 1 := 0;
  signal wr_data_c : std_ulogic_vector(c_lane_width - 1 downto 0);

  type flag_arr_t is array (0 to g_max_kernel_size - 1) of std_ulogic;
  signal in_frame_now : flag_arr_t := (others => '0');

  -- Which kernel-row tap (if any) each bank currently represents, for the
  -- capture stage's tap-index decode. S7: registered (maintained by
  -- 'control' together with 'out_row_q', see the geometry block below)
  -- rather than recomputed combinationally from 'row_top' every cycle.
  type kr_arr_t is array (0 to g_max_kernel_size - 1) of
    integer range 0 to g_max_kernel_size - 1;
  signal kr_of_q : kr_arr_t := (others => 0);

  -- One cycle behind 'in_frame_now'/'kr_of_q'/'kc_q' -- aligned with
  -- 'bank_rd_data', which lags the address by one registered read.
  signal in_frame_capture_q : flag_arr_t := (others => '0');
  signal kr_capture_q : kr_arr_t := (others => 0);
  signal kc_capture_q : unsigned(7 downto 0) := (others => '0');
  -- '1' the cycle after any cycle 'reading_q' was high -- i.e. this
  -- cycle's 'bank_rd_data' is meaningful and should be captured.
  signal capture_valid_q : std_ulogic := '0';

  -- Column walk position, 0 .. kernel_w_q (inclusive: 'kernel_w_q' itself
  -- is the one extra cycle needed to let the last column's registered
  -- read land -- see 'walk_control's comment). Bounded by a *registered*
  -- 'kernel_w_q' compare, not a variable-bound 'for' loop (the latter
  -- crashes GHDL's synthesis backend -- see the 'control' process' D11
  -- comment for the identical, already-hit issue).
  signal kc_q : unsigned(7 downto 0) := (others => '0');
  -- '1' while a column walk (read side) is in progress for the window at
  -- 'out_row_q'/'out_col_q'/'rd_tile_q'.
  signal reading_q : std_ulogic := '0';

  -- Tap-assembly register: accumulates one full window's taps across the
  -- 'kc' walk, then is presented as 'm_window_m2s.data' once
  -- 'window_valid' is asserted. Cleared to all-zero at the start of every
  -- walk, so taps that are out-of-frame (padding) or beyond the runtime
  -- 'kh'/'kw' simply stay '0' without being written -- same semantics as
  -- the predecessor combinational 'data_i'.
  signal assembly_q : tap_array_t(0 to c_window_data_length - 1) :=
    (others => (others => '0'));

  signal fire : std_ulogic;
  -- Registered: '1' once the 'kc' walk for the current window has fully
  -- landed. Unlike the predecessor design this is *not* a same-cycle
  -- function of 'row_ready' -- it trails it by the walk's pipeline
  -- latency (proposal doc section 7.1 item 4). Nothing downstream
  -- ('cnn_accel_pe_array', the testbench's scoreboard) depends on the
  -- old same-cycle timing; both only ever look for
  -- 'window_valid = 1 and m_window_s2m.ready = 1'.
  signal window_valid : std_ulogic := '0';
  signal consume : std_ulogic;
  signal last_pixel : std_ulogic;
  signal first_tile_flag, last_tile_flag : std_ulogic;
  -- Combinational: '1' once every real (unpadded) row/column this window
  -- needs has been fully written -- unchanged formula from the
  -- predecessor design (cnn_accel_window_gen_proposal.md section 4), just
  -- no longer wired directly to 'window_valid'. Only ever sampled from
  -- the 'idle' state below (registered state only, so no combinational
  -- loop through 's_stream_m2s.valid'/'m_window_s2m.ready').
  signal row_ready_i : std_ulogic;

  -- M7 correctness fix: combinational, '1' once the write side is about
  -- to advance into a physical row that would alias (same 'mod
  -- g_max_kernel_size' bank) the earliest real row 'out_row_q''s window
  -- still needs, before that whole output row (every 'out_col'/tile) has
  -- been consumed. Gates 's_stream_s2m.ready' below. Without this, the
  -- registered-read walk's lower read throughput (proposal doc section
  -- 7.1 item 4: ~'kw + 1' cycles/window versus the predecessor's
  -- same-cycle read) lets the write side, unblocked between 'reading_q'/
  -- 'window_valid' pulses, race more than 'g_max_kernel_size' physical
  -- rows ahead of 'out_row_q' -- wrapping the bank index back onto a row
  -- still pending read and silently corrupting it (caught by
  -- 'test_kernel_stride_shapes' et al: a slow 1x1-kernel consumer lets
  -- the writer outrun the reader by exactly 'g_max_kernel_size' rows).
  -- This is plain backpressure (freezes 's_stream_s2m.ready', the same
  -- knob the predecessor already used), not the out-of-scope
  -- double-buffering mitigation -- see the proposal doc section 8 risk 3.
  signal write_freeze_i : std_ulogic;

  -- '1' once a frame is in progress (between 'start' and the final
  -- window's acceptance); gates 's_stream_s2m.ready'/'window_valid' so
  -- nothing is accepted/emitted before the first 'start'.
  signal active_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Configuration, latched at 'start'.
  ------------------------------------------------------------------------

  signal kernel_h_q, kernel_w_q : unsigned(7 downto 0) := (others => '0');
  signal stride_h_q, stride_w_q : unsigned(7 downto 0) := (others => '0');
  signal pad_top_q, pad_left_q : unsigned(7 downto 0) := (others => '0');
  signal in_width_q, in_height_q : unsigned(15 downto 0) := (others => '0');
  signal out_width_q, out_height_q : unsigned(15 downto 0) := (others => '0');

  -- 'T = ceil(cfg_in_channels / g_tile_channels)', the number of
  -- input-channel tiles per output pixel (and per ingested input pixel).
  signal n_tiles_q : unsigned(15 downto 0) := to_unsigned(1, 16);
  -- Number of valid (real, non-zero-padded) channels in the *last* tile
  -- of the frame; equals 'g_tile_channels' when 'cfg_in_channels' is an
  -- exact multiple of it, 'cfg_in_channels mod g_tile_channels'
  -- otherwise. Only the last tile of a frame can ever be partial (a
  -- direct consequence of 'T' being a ceiling division) -- see D11 in
  -- the entity-level comment.
  signal last_tile_channels_q : natural range 0 to g_tile_channels := g_tile_channels;

  ------------------------------------------------------------------------
  -- Position counters.
  --
  -- 'cur_row_q'/'cur_col_q': the input pixel position currently being
  -- written (row/col within the raw, unpadded frame).
  -- 'wr_tile_q': the input-channel tile currently being written, within
  -- 'cur_col_q' ('0 .. n_tiles_q - 1').
  -- 'out_row_q'/'out_col_q': the next output window position to produce.
  -- 'rd_tile_q': the input-channel tile currently being read/emitted,
  -- within 'out_row_q'/'out_col_q' ('0 .. n_tiles_q - 1').
  ------------------------------------------------------------------------

  signal cur_row_q, cur_col_q : unsigned(15 downto 0) := (others => '0');
  signal wr_tile_q : unsigned(15 downto 0) := (others => '0');
  signal out_row_q, out_col_q : unsigned(15 downto 0) := (others => '0');
  signal rd_tile_q : unsigned(15 downto 0) := (others => '0');

  ------------------------------------------------------------------------
  -- S7: per-output-position geometry, precomputed into registers.
  --
  -- Every signal in this block is a function of 'out_row_q'/'out_col_q'/
  -- 'rd_tile_q' and the 'cfg_*' latch alone -- it changes once per output
  -- pixel (or per tile), never once per 'kc_q' cycle -- yet the
  -- predecessor design recomputed all of it combinationally *inside* the
  -- per-cycle read-address cone: two runtime multiplies, a runtime
  -- modulo, and four range compares per bank, all in series ahead of the
  -- BRAM address pins. That is what the logic-level histogram's ~966
  -- endpoints at levels 19-20 were.
  --
  -- These are maintained by 'control', by plain increments applied in the
  -- *same* clocked branch that advances 'out_row_q'/'out_col_q'/
  -- 'rd_tile_q'. That is deliberately not a pipeline stage: because they
  -- update in lockstep with the counters they are derived from, they are
  -- always exactly coherent with them and never one cycle stale. Two
  -- consequences worth spelling out:
  --   * 'row_ready_i' and 'write_freeze_i' keep their original,
  --     cycle-accurate semantics (they are handshake/interlock signals,
  --     not datapath -- see their declaration comments), so the M7
  --     bank-aliasing interlock argument carries over verbatim.
  --   * the read path gains NO latency: the walk's first address is still
  --     issued in the same cycle as before, so the 'kc_capture_q'/
  --     'kr_capture_q'/'in_frame_capture_q'/'capture_valid_q' one-deep
  --     delay pipeline and 'walk_control's single 'kc_q = kernel_w_q'
  --     drain cycle are unchanged.
  ------------------------------------------------------------------------

  -- Current-output-position geometry, all consumed by 'read_qualify' or
  -- by 'walk_control's walk-start load:
  --   'real_row_bot_q' = 'imin(row_top + kernel_h_q - 1, in_height_q - 1)'
  --   'has_real_row_q' = '(row_top <= in_height_q - 1)
  --                       and (row_top + kernel_h_q - 1 >= 0)'
  --   'anchor_limit_q' = 'imax(row_top, 0) + g_max_kernel_size' -- the
  --     'cur_row_q' value at which the write side would alias the
  --     earliest real row this output row still needs; pre-added so
  --     'write_freeze_i' is a single compare
  --   'row_ok_q(b)'    = 'kr_of_q(b) < kernel_h_q' and 'row_top +
  --     kr_of_q(b)' inside '[0, in_height_q - 1]' -- the row half of the
  --     former per-bank in-frame test
  --   'col_left_q'     = 'out_col_q * stride_w_q - pad_left_q'
  --   'real_col_right_q'/'has_real_col_q' -- the column analogues
  signal real_row_bot_q : coord_t := 0;
  signal has_real_row_q : std_ulogic := '0';
  signal anchor_limit_q : coord_t := g_max_kernel_size;
  signal row_ok_q : flag_arr_t := (others => '0');
  signal col_left_q : coord_t := 0;
  signal real_col_right_q : coord_t := 0;
  signal has_real_col_q : std_ulogic := '0';

  -- 'row_top mod g_max_kernel_size', kept as a ring counter (step
  -- 'stride_h_mod_q') so 'kr_of_q' never needs a runtime modulo of a
  -- 16-bit value by a non-power-of-two.
  signal row_top_mod_q : natural range 0 to g_max_kernel_size - 1 := 0;
  signal stride_h_mod_q : natural range 0 to g_max_kernel_size - 1 := 0;

  -- S7 second iteration, purely for timing: one-output-row / one-output-
  -- column *look-ahead* accumulators holding 'row_top', 'row_top +
  -- kernel_h_q - 1', 'row_top + g_max_kernel_size', 'col_left' and
  -- 'col_left + kernel_w_q - 1' for the position the counters are about
  -- to move to. Each is advanced by a single '+ stride' of its own, so
  -- the geometry registers above are updated by a *compare-and-select on
  -- register outputs* instead of by 'add, add, compare, select' in
  -- series. The first iteration of this rework left exactly that
  -- three-carry-chain next-state cone as the design's critical path
  -- (measured: 6.438 ns / 9 levels, 'row_top_q' -> 'row_ok_q',
  -- 154.23 MHz); splitting the adds out into these accumulators -- each
  -- of which is then its own single-chain register-to-register path --
  -- is what removes it.
  signal row_top_next_q, row_bot_next_q, row_top_k_next_q : coord_t := 0;
  signal col_left_next_q, col_right_next_q : coord_t := 0;

  -- Frame constants, one-shot at 'start': the '-1'/'-2' forms of the
  -- frame size that the range tests above compare against (so no test
  -- has to subtract first), the column-0 reload values copied into the
  -- column geometry at every output-row boundary, the per-kernel-row
  -- upper row bound 'in_height_q - 1 - kr' and the per-kernel-row
  -- 'kr < kernel_h_q' predicate.
  signal in_height_m1_q, in_width_m1_q, in_width_m2_q : coord_t := 0;
  signal col_left_start_q, col_left_start_next_q : coord_t := 0;
  signal col_right_start_next_q : coord_t := 0;
  signal real_col_right_start_q : coord_t := 0;
  signal has_real_col_start_q : std_ulogic := '0';
  type coord_arr_t is array (0 to g_max_kernel_size - 1) of coord_t;
  signal row_hi_bound_q : coord_arr_t := (others => 0);
  signal kr_valid_q : flag_arr_t := (others => '0');

  -- Row-bank cell address of the *first* column ('kc = 0') of the window
  -- at 'out_col_q'/'rd_tile_q': 'col_left * n_tiles_q + rd_tile_q', mod
  -- '2**c_addr_width'. Maintained by increments: '+1' per tile,
  -- '+ col_step_words_q' per output column, reloaded to
  -- 'row_start_words_q' at each output-row start.
  signal rd_base_q : word_t := (others => '0');
  -- One-shot at 'start' -- like the integer divisions in 'control', these
  -- are frame setup, not a per-cycle datapath. 'col_step_words_q' =
  -- 'stride_w_q * n_tiles_q - (n_tiles_q - 1)' (advance one output column
  -- *and* rewind the tile index to 0), 'row_start_words_q' =
  -- '(-pad_left_q) * n_tiles_q' (column 0 of a new output row),
  -- 'n_tiles_words_q' = 'n_tiles_q' (advance one 'kc' step); all mod
  -- '2**c_addr_width'.
  signal col_step_words_q : word_t := (others => '0');
  signal row_start_words_q : word_t := (others => '0');
  signal n_tiles_words_q : word_t := (others => '0');

  -- Read-address accumulator, advanced once per 'kc_q' step: as 'kc_q'
  -- increments by 1 'input_col' increments by 1, so the cell address
  -- increments by exactly 'n_tiles_q'. 'rd_word_q' is the raw modular
  -- accumulator; 'rd_addr_q' above is that value clamped to a legal bank
  -- address (0 while out of frame, exactly the predecessor's behaviour)
  -- so it can index 'bank_mem' straight out of a flip-flop.
  -- 'rd_col_q'/'rd_col_ok_q' are the matching 'input_col' accumulator and
  -- its '[0, in_width_q - 1]' range test -- registered too, so
  -- 'in_frame_now' is an AND of register outputs rather than the tail of
  -- the address cone.
  signal rd_word_q : word_t := (others => '0');
  signal rd_col_q : coord_t := 0;
  signal rd_col_ok_q : std_ulogic := '0';

  function imin(a, b : integer) return integer is
  begin
    if a < b then
      return a;
    else
      return b;
    end if;
  end function;

  function imax(a, b : integer) return integer is
  begin
    if a > b then
      return a;
    else
      return b;
    end if;
  end function;

  function to_sl(cond : boolean) return std_ulogic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

begin

  assert g_max_kernel_size >= 2
    report "cnn_accel_window_gen: g_max_kernel_size must be >= 2"
    severity failure;

  assert g_tile_channels * 8 <= axi_stream_data_sz
    report "cnn_accel_window_gen: g_tile_channels * 8 must be <= axi_stream_data_sz " &
      "(128) -- one channel-tile must fit in one s_stream beat"
    severity failure;

  ------------------------------------------------------------------------
  fire <= s_stream_m2s.valid and s_stream_s2m.ready;
  consume <= window_valid and m_window_s2m.ready;

  ------------------------------------------------------------------------
  -- Row-bank write data decode (combinational). The bank number and cell
  -- address it used to compute here are now the 'wr_bank_q'/'wr_addr_q'
  -- counters maintained by 'control' (S7 -- see their declaration
  -- comment); only the write *data* is still decoded combinationally,
  -- since it is a byte-lane mask on 's_stream_m2s.data' feeding DI, not a
  -- multiply feeding an address port.
  --
  -- D11: zero-pad the unused high channel lanes of a partial final tile,
  -- deterministically -- not whatever 's_stream_m2s.data' happens to
  -- carry there. Non-final tiles, and an exactly-dividing final tile
  -- ('last_tile_channels_q = g_tile_channels', making this a null loop
  -- range), are written verbatim. The loop range is constant
  -- (0 .. g_tile_channels - 1) with the runtime bound applied as a
  -- per-lane condition inside, rather than the more direct 'for c in
  -- last_tile_channels_q to ...'. Both are identical in simulation, but a
  -- variable loop range is not synthesizable: it crashes GHDL's synthesis
  -- backend ("limits of range are not constant", then an Ada assertion in
  -- synth-vhdl_expr.adb). Caught by this module's netlist build -- see
  -- module_cnn_accel.py get_build_projects().
  --
  -- NOTE: explicit sensitivity list, not 'process(all)' -- GHDL 7.0.0-dev's
  -- 'all' inference does not reliably track signals read only through
  -- nested loops/array indexing.
  ------------------------------------------------------------------------
  wr_decode : process(
    wr_tile_q, n_tiles_q, last_tile_channels_q, s_stream_m2s.data
  )
    variable v_write_data : std_ulogic_vector(c_lane_width - 1 downto 0);
  begin
    v_write_data := s_stream_m2s.data(c_lane_width - 1 downto 0);
    if wr_tile_q = n_tiles_q - 1 then
      for c in 0 to g_tile_channels - 1 loop
        if c >= last_tile_channels_q then
          v_write_data(8 * (c + 1) - 1 downto 8 * c) := (others => '0');
        end if;
      end loop;
    end if;
    wr_data_c <= v_write_data;
  end process;

  ------------------------------------------------------------------------
  -- Row banks: 'g_max_kernel_size' physically independent memories (see
  -- the 'row_banks' comment above), one 'bank_mem' per generate branch.
  -- Each branch is the exact same shape as
  -- cnn_accel_weight_buffer.vhd's 'weight_mem'/'read_ports' pair: one
  -- statically-addressed write process (write enable = 'fire' and this
  -- branch's bank number matching 'wr_bank_q'), and one registered,
  -- unconditional, single read process (the shared 'rd_addr_q' is only
  -- meaningful while 'in_frame_now(b)' is set, checked at capture time,
  -- not here) --
  -- no reset on 'bank_mem' itself, same as cnn_accel_weight_buffer.vhd.
  ------------------------------------------------------------------------
  gen_banks : for b in 0 to g_max_kernel_size - 1 generate
    signal bank_mem : row_bank_t;
  begin

    write_port : process(clk)
    begin
      if rising_edge(clk) then
        if fire = '1' and wr_bank_q = b then
          bank_mem(wr_addr_q) <= wr_data_c;
        end if;
      end if;
    end process;

    read_port : process(clk)
    begin
      if rising_edge(clk) then
        bank_rd_data(b) <= bank_mem(rd_addr_q);
      end if;
    end process;

  end generate;

  -- Freeze further input acceptance whenever a window is pending (its
  -- 'kc' walk in progress, 'reading_q') or has landed and not yet been
  -- consumed ('window_valid') -- a row bank must not be overwritten until
  -- every tap that still needs it has been read out. Both 'reading_q'
  -- and 'window_valid' depend only on registered state (never on
  -- 's_stream_m2s.valid'/'m_window_s2m.ready'), so this has no
  -- combinational loop.
  s_stream_s2m.ready <= active_q and not write_freeze_i and
    (not (window_valid or reading_q) or (window_valid and m_window_s2m.ready));

  ------------------------------------------------------------------------
  -- Configuration latch, position counters, write-side address counters,
  -- and the S7 per-output-position geometry registers (see their
  -- declaration block above -- they are updated here, in the very same
  -- clocked branches that advance 'out_row_q'/'out_col_q'/'rd_tile_q',
  -- which is what keeps them exactly coherent with those counters rather
  -- than a cycle behind them).
  ------------------------------------------------------------------------
  control : process(clk)
    variable v_num_w, v_num_h : integer;
    variable v_in_channels, v_n_tiles : integer;
    variable v_kh, v_kw, v_sh, v_sw, v_pt, v_pl, v_inh, v_inw : integer;
    variable v_row_top, v_col_left : coord_t;
    variable v_mod : natural range 0 to 2 * g_max_kernel_size - 2;
    variable v_kr : natural range 0 to 2 * g_max_kernel_size - 1;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        active_q <= '0';
        cur_row_q <= (others => '0');
        cur_col_q <= (others => '0');
        wr_tile_q <= (others => '0');
        out_row_q <= (others => '0');
        out_col_q <= (others => '0');
        rd_tile_q <= (others => '0');
        wr_bank_q <= 0;
        wr_addr_q <= 0;

      elsif start = '1' then
        assert unsigned(cfg_kernel_h) <= to_unsigned(g_max_kernel_size, 8)
          and unsigned(cfg_kernel_w) <= to_unsigned(g_max_kernel_size, 8)
          report "cnn_accel_window_gen: cfg_kernel_h/w must be <= g_max_kernel_size"
          severity failure;

        kernel_h_q <= unsigned(cfg_kernel_h);
        kernel_w_q <= unsigned(cfg_kernel_w);
        stride_h_q <= unsigned(cfg_stride_h);
        stride_w_q <= unsigned(cfg_stride_w);
        pad_top_q <= unsigned(cfg_pad_top);
        pad_left_q <= unsigned(cfg_pad_left);
        in_width_q <= unsigned(cfg_in_width);
        in_height_q <= unsigned(cfg_in_height);

        -- out_dim = (in_dim + pad_lo + pad_hi - kernel) / stride + 1.
        -- One-shot per 'start' (not a per-cycle datapath), so plain
        -- integer division here is deliberate -- see
        -- cnn_accel_window_gen_proposal.md section 5.
        v_num_w := to_integer(unsigned(cfg_in_width)) + to_integer(unsigned(cfg_pad_left))
          + to_integer(unsigned(cfg_pad_right)) - to_integer(unsigned(cfg_kernel_w));
        v_num_h := to_integer(unsigned(cfg_in_height)) + to_integer(unsigned(cfg_pad_top))
          + to_integer(unsigned(cfg_pad_bottom)) - to_integer(unsigned(cfg_kernel_h));

        out_width_q <= to_unsigned(v_num_w / to_integer(unsigned(cfg_stride_w)) + 1, 16);
        out_height_q <= to_unsigned(v_num_h / to_integer(unsigned(cfg_stride_h)) + 1, 16);

        -- T = ceil(in_channels / g_tile_channels); only the last tile of
        -- the frame can be partial (see 'last_tile_channels_q's comment).
        v_in_channels := to_integer(unsigned(cfg_in_channels));
        v_n_tiles := (v_in_channels + g_tile_channels - 1) / g_tile_channels;

        assert to_integer(unsigned(cfg_in_width)) * v_n_tiles <= g_max_row_tile_words
          report "cnn_accel_window_gen: cfg_in_width * ceil(cfg_in_channels/g_tile_channels) " &
            "must be <= g_max_row_tile_words"
          severity failure;

        n_tiles_q <= to_unsigned(v_n_tiles, 16);
        last_tile_channels_q <= v_in_channels - (v_n_tiles - 1) * g_tile_channels;

        cur_row_q <= (others => '0');
        cur_col_q <= (others => '0');
        wr_tile_q <= (others => '0');
        out_row_q <= (others => '0');
        out_col_q <= (others => '0');
        rd_tile_q <= (others => '0');
        active_q <= '1';

        wr_bank_q <= 0;
        wr_addr_q <= 0;

        -- S7: seed every geometry register / address accumulator for
        -- 'out_row = out_col = rd_tile = 0'. Read from the 'cfg_*' ports
        -- rather than from 'kernel_h_q' et al, which are only being
        -- latched this same cycle.
        v_kh := to_integer(unsigned(cfg_kernel_h));
        v_kw := to_integer(unsigned(cfg_kernel_w));
        v_sh := to_integer(unsigned(cfg_stride_h));
        v_sw := to_integer(unsigned(cfg_stride_w));
        v_pt := to_integer(unsigned(cfg_pad_top));
        v_pl := to_integer(unsigned(cfg_pad_left));
        v_inh := to_integer(unsigned(cfg_in_height));
        v_inw := to_integer(unsigned(cfg_in_width));

        v_row_top := -v_pt;
        v_col_left := -v_pl;

        -- Frame constants in the exact '-1'/'-2' forms the range tests
        -- below compare against, so no test has to subtract first.
        in_height_m1_q <= v_inh - 1;
        in_width_m1_q <= v_inw - 1;
        in_width_m2_q <= v_inw - 2;

        real_row_bot_q <= imin(v_row_top + v_kh - 1, v_inh - 1);
        has_real_row_q <= to_sl(v_row_top <= v_inh - 1 and v_row_top + v_kh - 1 >= 0);
        anchor_limit_q <= imax(v_row_top, 0) + g_max_kernel_size;

        -- Row look-ahead accumulators, seeded for output row 1.
        row_top_next_q <= v_row_top + v_sh;
        row_bot_next_q <= v_row_top + v_sh + v_kh - 1;
        row_top_k_next_q <= v_row_top + v_sh + g_max_kernel_size;

        stride_h_mod_q <= v_sh mod g_max_kernel_size;
        v_mod := (-v_pt) mod g_max_kernel_size;
        row_top_mod_q <= v_mod;
        for b in 0 to g_max_kernel_size - 1 loop
          v_kr := b + g_max_kernel_size - v_mod;
          if v_kr >= g_max_kernel_size then
            v_kr := v_kr - g_max_kernel_size;
          end if;
          kr_of_q(b) <= v_kr;
          row_ok_q(b) <= to_sl(
            v_kr < v_kh and v_row_top + v_kr >= 0 and v_row_top + v_kr <= v_inh - 1
          );
        end loop;

        -- Per-kernel-row upper bound and validity, so the per-output-row
        -- update of 'row_ok_q' is a compare of 'row_top_next_q' against a
        -- register rather than an add ('row_top + kr') followed by two
        -- compares: 'row_top + kr <= in_height - 1' is exactly
        -- 'row_top <= row_hi_bound_q(kr)'.
        for k in 0 to g_max_kernel_size - 1 loop
          row_hi_bound_q(k) <= v_inh - 1 - k;
          kr_valid_q(k) <= to_sl(k < v_kh);
        end loop;

        col_left_q <= v_col_left;
        real_col_right_q <= imin(v_col_left + v_kw - 1, v_inw - 1);
        has_real_col_q <= to_sl(v_col_left <= v_inw - 1 and v_col_left + v_kw - 1 >= 0);

        -- Column look-ahead accumulators, seeded for output column 1.
        col_left_next_q <= v_col_left + v_sw;
        col_right_next_q <= v_col_left + v_sw + v_kw - 1;

        -- Column-0 reload values: what the column geometry and its
        -- look-ahead must become at every output-row boundary. Registered
        -- once here so that boundary is a set of plain register copies.
        col_left_start_q <= v_col_left;
        real_col_right_start_q <= imin(v_col_left + v_kw - 1, v_inw - 1);
        has_real_col_start_q <= to_sl(
          v_col_left <= v_inw - 1 and v_col_left + v_kw - 1 >= 0
        );
        col_left_start_next_q <= v_col_left + v_sw;
        col_right_start_next_q <= v_col_left + v_sw + v_kw - 1;

        -- The only multiplies left in this entity, and all three are
        -- one-shot frame setup (same argument as the divisions above).
        n_tiles_words_q <= to_unsigned(v_n_tiles mod 2 ** c_addr_width, c_addr_width);
        col_step_words_q <= to_unsigned(
          (v_sw * v_n_tiles - v_n_tiles + 1) mod 2 ** c_addr_width, c_addr_width
        );
        row_start_words_q <= to_unsigned(
          (-v_pl * v_n_tiles) mod 2 ** c_addr_width, c_addr_width
        );
        rd_base_q <= to_unsigned((-v_pl * v_n_tiles) mod 2 ** c_addr_width, c_addr_width);

      else
        if fire = '1' then
          -- Row-bank write itself ('wr_bank_q'/'wr_addr_q'/'wr_data_c')
          -- happens in the 'gen_banks' generate block above, not here --
          -- see the 'row_banks' comment above for why. This process only
          -- advances the write-side position counters, which since S7
          -- includes the bank/cell address the write actually uses: one
          -- accepted beat is always the next cell of the current input
          -- row, and an input-row boundary restarts at cell 0 of the next
          -- bank in the ring.
          if wr_tile_q = n_tiles_q - 1 then
            wr_tile_q <= (others => '0');
            if cur_col_q = in_width_q - 1 then
              cur_col_q <= (others => '0');
              cur_row_q <= cur_row_q + 1;
              wr_addr_q <= 0;
              if wr_bank_q = g_max_kernel_size - 1 then
                wr_bank_q <= 0;
              else
                wr_bank_q <= wr_bank_q + 1;
              end if;
            else
              cur_col_q <= cur_col_q + 1;
              wr_addr_q <= wr_addr_q + 1;
            end if;
          else
            wr_tile_q <= wr_tile_q + 1;
            wr_addr_q <= wr_addr_q + 1;
          end if;
        end if;

        if consume = '1' then
          if rd_tile_q = n_tiles_q - 1 then
            rd_tile_q <= (others => '0');

            if out_col_q = out_width_q - 1 then
              out_col_q <= (others => '0');

              -- Output-row boundary: column geometry, and its look-ahead,
              -- back to the column-0 values registered at 'start'. All
              -- plain register-to-register copies -- no arithmetic.
              col_left_q <= col_left_start_q;
              real_col_right_q <= real_col_right_start_q;
              has_real_col_q <= has_real_col_start_q;
              col_left_next_q <= col_left_start_next_q;
              col_right_next_q <= col_right_start_next_q;
              rd_base_q <= row_start_words_q;

              if out_row_q /= out_height_q - 1 then
                out_row_q <= out_row_q + 1;

                -- ... and row geometry one stride further down. Every
                -- value needed here already sits in a look-ahead
                -- accumulator, so this branch is a copy plus at most one
                -- compare -- never the 'add, add, compare, select' series
                -- that was the first S7 iteration's critical path.
                -- 'anchor_limit' clamps 'row_top' up to 0 (top padding
                -- reserves no bank), which is a sign-bit select here.
                real_row_bot_q <= imin(row_bot_next_q, in_height_m1_q);
                has_real_row_q <= to_sl(
                  row_top_next_q <= in_height_m1_q and row_bot_next_q >= 0
                );
                if row_top_next_q >= 0 then
                  anchor_limit_q <= row_top_k_next_q;
                else
                  anchor_limit_q <= g_max_kernel_size;
                end if;

                -- Push the look-ahead one output row further. Each of the
                -- three is its own single '+ stride' carry chain, from
                -- flip-flop to flip-flop and nothing else.
                row_top_next_q <= row_top_next_q + to_integer(stride_h_q);
                row_bot_next_q <= row_bot_next_q + to_integer(stride_h_q);
                row_top_k_next_q <= row_top_k_next_q + to_integer(stride_h_q);

                v_mod := row_top_mod_q + stride_h_mod_q;
                if v_mod >= g_max_kernel_size then
                  v_mod := v_mod - g_max_kernel_size;
                end if;
                row_top_mod_q <= v_mod;

                for b in 0 to g_max_kernel_size - 1 loop
                  v_kr := b + g_max_kernel_size - v_mod;
                  if v_kr >= g_max_kernel_size then
                    v_kr := v_kr - g_max_kernel_size;
                  end if;
                  kr_of_q(b) <= v_kr;
                  row_ok_q(b) <= kr_valid_q(v_kr) and to_sl(
                    row_top_next_q >= -v_kr
                    and row_top_next_q <= row_hi_bound_q(v_kr)
                  );
                end loop;
              end if;
            else
              out_col_q <= out_col_q + 1;

              -- Same shape as the row advance above: copy out of the
              -- look-ahead accumulators, then push those one stride on.
              col_left_q <= col_left_next_q;
              real_col_right_q <= imin(col_right_next_q, in_width_m1_q);
              has_real_col_q <= to_sl(
                col_left_next_q <= in_width_m1_q and col_right_next_q >= 0
              );

              col_left_next_q <= col_left_next_q + to_integer(stride_w_q);
              col_right_next_q <= col_right_next_q + to_integer(stride_w_q);

              rd_base_q <= rd_base_q + col_step_words_q;
            end if;

            if last_pixel = '1' then
              active_q <= '0';
            end if;
          else
            rd_tile_q <= rd_tile_q + 1;
            rd_base_q <= rd_base_q + 1;
          end if;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  last_pixel <= '1' when (out_row_q = out_height_q - 1 and out_col_q = out_width_q - 1) else '0';
  first_tile_flag <= '1' when rd_tile_q = 0 else '0';
  last_tile_flag <= '1' when rd_tile_q = n_tiles_q - 1 else '0';
  done <= consume and last_pixel and last_tile_flag;

  m_window_m2s.valid <= window_valid;
  m_window_m2s.last <= last_pixel and last_tile_flag;
  m_window_m2s.first_tile <= first_tile_flag;
  m_window_m2s.last_tile <= last_tile_flag;
  m_window_m2s.data <= assembly_q;

  ------------------------------------------------------------------------
  -- Read-side qualification (combinational). S7: this is all that is left
  -- of the predecessor's 'addr_gen' process. The read *address* itself is
  -- no longer computed here at all -- it is the 'rd_addr_q' accumulator
  -- driven by 'walk_control' below -- and every per-output-position term
  -- ('row_top', 'col_left', 'has_real_row', 'real_row_bot', the anchor
  -- row, 'kr_of_q(b)', and the whole row half of the per-bank in-frame
  -- test) is now a register maintained by 'control'. What remains is
  -- three shallow tests against register outputs, all of them ending on
  -- flip-flops rather than on a BRAM address port:
  --
  --   * 'row_ready_i': the unchanged spatial readiness test from the
  --     predecessor design (cnn_accel_window_gen_proposal.md section 4) --
  --     ready once the largest real row/column this window needs has been
  --     fully written: no real row at all (window entirely vertical
  --     padding) -> ready immediately; the needed row already complete
  --     ('cur_row_q > real_row_bot') -> ready regardless of columns; still
  --     writing that exact row -> also need its columns caught up (or no
  --     real column at all). Same formula, same cycle: 'has_real_row_q'/
  --     'real_row_bot_q'/'has_real_col_q'/'real_col_right_q' are updated
  --     in lockstep with 'out_row_q'/'out_col_q', so this is bit- and
  --     cycle-identical to the predecessor's inline arithmetic, not a
  --     delayed approximation of it. That matters: 'row_ready_i' is NOT
  --     monotone in 'out_col_q' (advancing an output column raises
  --     'real_col_right'), so a merely *registered* version of it could
  --     read '1' for a window whose columns are not written yet and start
  --     a walk too early. Coherent-by-construction avoids that entirely.
  --
  --   * 'write_freeze_i': likewise unchanged, and for the same reason --
  --     'anchor_limit_q' is 'imax(row_top, 0) + g_max_kernel_size'
  --     maintained in lockstep with 'out_row_q', so the M7 bank-aliasing
  --     interlock (see 'write_freeze_i's declaration comment) asserts and
  --     releases on exactly the same cycles as before. 'row_top' clamped
  --     up to 0 because a negative 'row_top' (top padding) reserves no
  --     bank at all: there is no row -2 to protect, so the first bank
  --     actually in use starts at row 0.
  --
  --   * 'in_frame_now(b)': an AND of four register outputs plus one
  --     8-bit compare. 'row_ok_q(b)' carries 'kr_of_q(b) < kernel_h_q'
  --     and 'row_top + kr_of_q(b)' in '[0, in_height_q - 1]';
  --     'rd_col_ok_q' carries 'input_col' in '[0, in_width_q - 1]' for
  --     the column the *currently presented* address reads (it is
  --     advanced in lockstep with 'rd_addr_q', so the two are always
  --     aligned). 'kc_q < kernel_w_q' excludes the drain cycle.
  --
  -- NOTE: explicit sensitivity list, not 'process(all)' -- GHDL 7.0.0-dev's
  -- 'all' inference does not reliably track signals read only through
  -- nested loops/array indexing.
  ------------------------------------------------------------------------
  read_qualify : process(
    kernel_w_q, cur_row_q, cur_col_q, kc_q, reading_q, rd_col_ok_q,
    has_real_row_q, real_row_bot_q, has_real_col_q, real_col_right_q,
    anchor_limit_q, row_ok_q
  )
    variable v_cur_row : coord_t;
    variable v_kc_real : std_ulogic;
  begin
    v_cur_row := to_integer(cur_row_q);

    if has_real_row_q = '0'
      or v_cur_row > real_row_bot_q
      or (
        v_cur_row = real_row_bot_q
        and (has_real_col_q = '0' or to_integer(cur_col_q) > real_col_right_q)
      )
    then
      row_ready_i <= '1';
    else
      row_ready_i <= '0';
    end if;

    if has_real_row_q = '1' and v_cur_row >= anchor_limit_q then
      write_freeze_i <= '1';
    else
      write_freeze_i <= '0';
    end if;

    v_kc_real := to_sl(kc_q < kernel_w_q);

    for b in 0 to g_max_kernel_size - 1 loop
      in_frame_now(b) <= reading_q and v_kc_real and row_ok_q(b) and rd_col_ok_q;
    end loop;
  end process;

  ------------------------------------------------------------------------
  -- Column walk (read side) + tap-assembly capture. Every cycle:
  --
  -- 1. Capture stage: if 'capture_valid_q' (last cycle issued a real
  --    read address), pack 'bank_rd_data' -- now valid, one cycle after
  --    that address -- into 'assembly_q' at tap slot
  --    'kr_capture_q(b) * kernel_w_q + kc_capture_q', for whichever banks
  --    were actually in-frame. 'tap_idx' depends on the runtime 'kw', so
  --    it cannot index 'assembly_q' directly here without becoming a
  --    decoder anyway; the destination slot is selected with a
  --    constant-bound loop, identical in simulation, a mux in hardware --
  --    the same technique the predecessor combinational process used
  --    for the same reason (runtime tap index into a fixed-size
  --    aggregate).
  -- 2. Walk/valid state machine: 'window_valid' (idle) -> 'reading_q'
  --    (kc = 0 .. kernel_w_q, one column issued per cycle) ->
  --    'window_valid' again once the last column's registered read has
  --    landed (kc_q = kernel_w_q is one extra "drain" cycle: the
  --    address for column 'kernel_w_q - 1' was issued the cycle before,
  --    its data is captured this cycle). 'window_valid' only asserts
  --    from 'idle' when 'row_ready_i' is true, i.e. every input pixel
  --    the window needs has already been written -- unchanged invariant
  --    from the predecessor design, just no longer same-cycle (proposal
  --    doc section 7.1 item 4 / section 8 risk 3).
  -- 3. S7: the read-address accumulators. 'rd_word_q'/'rd_col_q' are
  --    loaded -- from registers 'control' keeps coherent with
  --    'out_col_q'/'rd_tile_q' -- in the *same* cycle the walk is armed,
  --    i.e. the cycle that already set 'reading_q'/'kc_q', so the
  --    'kc = 0' address is presented on exactly the cycle the
  --    predecessor's combinational 'addr_gen' presented it. The read
  --    latency is therefore still one cycle and items 1 and 2 above are
  --    untouched: the delay pipeline stays one deep and the drain stays
  --    one cycle. Each subsequent 'kc' step is '+ n_tiles_words_q' (one
  --    input column = 'n_tiles_q' consecutive channel-tile cells),
  --    replacing the predecessor's 'input_col * n_tiles_q + rd_tile_q'
  --    multiply. 'rd_addr_q' is that accumulator clamped to a legal
  --    address, so the BRAM address port is driven by a flip-flop and
  --    nothing else; the clamp condition is the same
  --    '[0, in_width_q - 1]' test that gates 'in_frame_now', so an
  --    out-of-frame cycle reads cell 0 exactly as the predecessor's
  --    'rd_addr(b) <= 0' branch did.
  ------------------------------------------------------------------------
  walk_control : process(clk)
    variable v_next_word : word_t;
    variable v_next_col : coord_t;
    variable v_next_ok : boolean;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        reading_q <= '0';
        window_valid <= '0';
        kc_q <= (others => '0');
        capture_valid_q <= '0';
        kc_capture_q <= (others => '0');
        in_frame_capture_q <= (others => '0');
        kr_capture_q <= (others => 0);
        assembly_q <= (others => (others => '0'));
        rd_addr_q <= 0;
        rd_word_q <= (others => '0');
        rd_col_q <= 0;
        rd_col_ok_q <= '0';

      elsif start = '1' then
        reading_q <= '0';
        window_valid <= '0';
        kc_q <= (others => '0');
        capture_valid_q <= '0';
        kc_capture_q <= (others => '0');
        in_frame_capture_q <= (others => '0');
        kr_capture_q <= (others => 0);
        assembly_q <= (others => (others => '0'));
        rd_addr_q <= 0;
        rd_word_q <= (others => '0');
        rd_col_q <= 0;
        rd_col_ok_q <= '0';

      else
        -- Delay pipeline: always shifts by one cycle, so the capture
        -- stage below sees last cycle's issued column/in-frame/bank
        -- mapping alongside this cycle's now-valid 'bank_rd_data'.
        capture_valid_q <= reading_q;
        kc_capture_q <= kc_q;
        kr_capture_q <= kr_of_q;
        in_frame_capture_q <= in_frame_now;

        if capture_valid_q = '1' then
          for b in 0 to g_max_kernel_size - 1 loop
            if in_frame_capture_q(b) = '1' then
              for t in 0 to g_max_kernel_size * g_max_kernel_size - 1 loop
                if t = kr_capture_q(b) * to_integer(kernel_w_q) + to_integer(kc_capture_q) then
                  for c in 0 to g_tile_channels - 1 loop
                    assembly_q(t * g_tile_channels + c) <= bank_rd_data(b)(8 * (c + 1) - 1 downto 8 * c);
                  end loop;
                end if;
              end loop;
            end if;
          end loop;
        end if;

        if window_valid = '1' then
          if consume = '1' then
            window_valid <= '0';
          end if;

        elsif reading_q = '1' then
          if kc_q = kernel_w_q then
            reading_q <= '0';
            window_valid <= '1';
          else
            kc_q <= kc_q + 1;

            v_next_word := rd_word_q + n_tiles_words_q;
            v_next_col := rd_col_q + 1;
            -- '(rd_col_q + 1) in [0, in_width - 1]' expressed on
            -- 'rd_col_q' itself, so the range test does not wait on the
            -- increment's carry chain: registered bounds, one compare.
            v_next_ok := rd_col_q >= -1 and rd_col_q <= in_width_m2_q;

            rd_word_q <= v_next_word;
            rd_col_q <= v_next_col;
            rd_col_ok_q <= to_sl(v_next_ok);
            if v_next_ok then
              rd_addr_q <= to_integer(v_next_word);
            else
              rd_addr_q <= 0;
            end if;
          end if;

        else
          -- Idle: start a new walk once this window's inputs are fully
          -- written. 'assembly_q' is cleared here (not just relying on
          -- its reset value) so a tap left out-of-frame this window
          -- (fewer real rows/columns than 'g_max_kernel_size', or
          -- padding) reads back '0' rather than a previous window's
          -- stale value.
          if active_q = '1' and row_ready_i = '1' then
            reading_q <= '1';
            kc_q <= (others => '0');
            assembly_q <= (others => (others => '0'));

            v_next_ok := col_left_q >= 0 and col_left_q <= in_width_m1_q;
            rd_word_q <= rd_base_q;
            rd_col_q <= col_left_q;
            rd_col_ok_q <= to_sl(v_next_ok);
            if v_next_ok then
              rd_addr_q <= to_integer(rd_base_q);
            else
              rd_addr_q <= 0;
            end if;
          end if;
        end if;
      end if;
    end if;
  end process;

end architecture a;
