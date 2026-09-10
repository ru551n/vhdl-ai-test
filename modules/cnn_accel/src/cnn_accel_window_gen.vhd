library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Configurable K_h x K_w / stride / padding (with a configurable pad
-- value, ISA v2.1) / input-channel-tiled sliding-window generator. See modules/cnn_accel/doc/cnn_accel_window_gen_req.md,
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
    g_tile_channels : positive;
    -- Number of tap-assembly buffers a window can be built into ("N"
    -- below). 1 is the historical single-buffered behaviour; >= 2 lets
    -- the next window's column walk overlap the current window's
    -- presentation, which is what turns the read side from
    -- 'kernel_w + 3' cycles per window beat into 'max(kernel_w,
    -- (kernel_w + 2) / N)' -- see the 'assembly_q' declaration and the
    -- 'walk_control' comment for the pipeline that makes that true, and
    -- the 's_stream_s2m.ready' comment for the row-bank invariant that
    -- has to hold with more than one window in flight.
    --
    -- Cost: one full 'g_max_kernel_size**2 * g_tile_channels'-byte tap
    -- register bank per buffer, plus an N:1 output mux of the same
    -- width. That is why this is a generic and not a constant: the CONV
    -- instance is throughput-critical and pays it, the POOL instance
    -- (K = 5, so 200 bytes per buffer) is not and does not.
    g_assembly_buffers : positive := 1
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
    -- ISA v2.1: the signed int8 value every PADDED tap of the window
    -- takes -- the input tensor's quantization zero-point, not
    -- necessarily 0. Defaults to zero, which is the pre-v2.1
    -- zero-padding behaviour exactly; 'cnn_accel_top' ties the CONV
    -- instance's input to zero deliberately and drives only the POOL
    -- instance's from the descriptor. See the 'assembly_q' comment for
    -- where it lands: an all-'pad value' clear at the start of every
    -- column walk, so a tap that is never written (out of frame, or
    -- beyond the runtime kernel size) reads back as padding.
    cfg_pad_value : in std_ulogic_vector(7 downto 0) := (others => '0');
    cfg_in_width : in std_ulogic_vector(15 downto 0);
    cfg_in_height : in std_ulogic_vector(15 downto 0);
    cfg_in_channels : in std_ulogic_vector(15 downto 0);
    -- Output frame dimensions for the command being started, i.e. exactly
    -- '(in_dim + pad_lo + pad_hi - kernel) / stride + 1' in each axis.
    --
    -- This module used to derive them itself, in the single cycle in which
    -- 'start' latched the configuration. Two runtime divisions by
    -- 'cfg_stride_w'/'cfg_stride_h' in one combinational cone off
    -- 'cnn_accel_cmd_proc's descriptor register is what made
    -- 'out_height_q' a 135-logic-level endpoint and cost the first
    -- top-level build 47 ns of setup slack at 150 MHz. 'cnn_accel_cmd_proc'
    -- already computes the identical quantity, once per command, on its
    -- own 16-step restoring divider ('st_div_w'/'st_div_h', reaching
    -- 'out_w_q'/'out_h_q' several states before it pulses 'start'), so
    -- the division is not repeated here -- it is received.
    --
    -- Contract: stable and correct for the command's geometry whenever
    -- 'start' is asserted. Both are '>= 1' for any geometry
    -- 'cnn_accel_cmd_proc' does not reject.
    cfg_out_width : in std_ulogic_vector(15 downto 0);
    cfg_out_height : in std_ulogic_vector(15 downto 0);
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

  -- 'kr_of_q(b) * kernel_w_q', i.e. the first tap index of the kernel row
  -- bank 'b' currently represents, maintained as a register alongside
  -- 'kr_of_q' itself.
  --
  -- Why it is not just computed in the capture stage (measured, Vivado
  -- xc7a200tfbg484-2, 500 MHz constraint, this entity at the POOL
  -- geometry -- g_max_kernel_size = 5, g_max_row_tile_words = 1920):
  -- 'kr_capture_q(b) * kernel_w_q + kc_capture_q' was the critical path
  -- at 7.166 ns / 9 logic levels / 3 CARRY4 -- 139.55 MHz, i.e. BELOW the
  -- 150 MHz target -- from 'kr_capture_q's output, through the runtime
  -- multiply's carry chain, into the write-enable decode of all
  -- 'g_max_kernel_size**2 * g_tile_channels' (200) 'assembly_q' lanes.
  -- The multiply is the part that does not belong there: 'kr_of_q' only
  -- ever changes at an output-row boundary, so the product can be
  -- maintained on that (slow, non-critical) path instead of being
  -- recomputed every cycle on the (fast, critical) one. What is left in
  -- the capture stage is 'kr_base + kc', one small add.
  --
  -- 'kr_kw_q' is the 'k * kernel_w' table this is selected from, computed
  -- once at 'start', so the per-output-row update is a mux over
  -- 'g_max_kernel_size' registers and not a second runtime multiply.
  type kr_base_arr_t is array (0 to g_max_kernel_size - 1) of
    integer range 0 to g_max_kernel_size * g_max_kernel_size - 1;
  signal kr_base_q : kr_base_arr_t := (others => 0);
  signal kr_kw_q : kr_base_arr_t := (others => 0);

  -- One cycle behind 'in_frame_now'/'kr_of_q'/'kc_q' -- aligned with
  -- 'bank_rd_data', which lags the address by one registered read.
  signal in_frame_capture_q : flag_arr_t := (others => '0');
  -- (There is no 'kr_capture_q': the capture stage needs only the tap
  -- base index, never the kernel-row number itself.)
  signal kr_base_capture_q : kr_base_arr_t := (others => 0);
  signal kc_capture_q : unsigned(7 downto 0) := (others => '0');

  -- The capture stage's tap slot, PRE-DECODED to one-hot and registered
  -- alongside the rest of the capture pipeline.
  --
  -- The capture write enable used to be
  -- 't = kr_base_capture_q(b) + to_integer(kc_capture_q)', i.e. an adder
  -- and a comparator per (row, tap) evaluated combinationally out of
  -- 'kc_capture_q' into every one of the assembly buffers' clock enables.
  -- Post-route 'kc_capture_q -> assembly_q[*]' was 224 of the design's
  -- worst 400 endpoints at 6-8 logic levels (the 5x5 pool instance; the
  -- 3x3 conv one is the same code, one third the width).
  --
  -- 'shared/TimingAndResources.md', Fundamentals, "Control structure":
  -- decode the index into one-bit registered flags once, at the point the
  -- index is latched, and have every site test its flag. The decode is
  -- computed from exactly the same 'kr_base_walk_q'/'kc_q' pair, one cycle
  -- earlier and in the same register stage as 'kc_capture_q' itself, so
  -- the selected slot is bit-identical; only its arrival time moves.
  type tap_sel_arr_t is array (0 to g_max_kernel_size - 1) of
    std_ulogic_vector(0 to g_max_kernel_size * g_max_kernel_size - 1);
  signal tap_sel_capture_q : tap_sel_arr_t := (others => (others => '0'));
  -- '1' the cycle after any cycle 'issue_q' was high -- i.e. this
  -- cycle's 'bank_rd_data' is meaningful and should be captured.
  signal capture_valid_q : std_ulogic := '0';
  -- '1' when the column being captured this cycle is the LAST column
  -- ('kc = kernel_w - 1') of its window: the buffer it lands in is
  -- complete at the end of this cycle and is marked full.
  signal capture_last_q : std_ulogic := '0';
  -- Which assembly buffer this cycle's capture belongs to (one cycle
  -- behind 'buf_issue_q', like every other signal in this pipeline).
  signal buf_capture_q : natural range 0 to g_assembly_buffers - 1 := 0;

  -- Column walk position, 0 .. kernel_w_q - 1. Unlike the predecessor
  -- design there is no extra 'kc_q = kernel_w_q' drain step: the drain is
  -- now implicit in the capture stage (the last column's registered read
  -- lands one cycle after its address was issued, and *that* cycle is
  -- what marks the buffer full), which is one of the two cycles per
  -- window this rework removes. Bounded by a *registered* 'kernel_w_q'
  -- compare, not a variable-bound 'for' loop (the latter crashes GHDL's
  -- synthesis backend -- see the 'control' process' D11 comment for the
  -- identical, already-hit issue).
  signal kc_q : unsigned(7 downto 0) := (others => '0');
  -- '1' while this cycle's 'rd_addr_q' is a real column address of a
  -- window being walked (the predecessor's 'reading_q', minus the drain
  -- cycle it also covered).
  signal issue_q : std_ulogic := '0';
  -- Which assembly buffer the walk currently issuing addresses is
  -- filling. Advances (mod 'g_assembly_buffers') at every walk start, so
  -- buffers are allocated strictly in launch order and, since windows are
  -- consumed in that same order, freed strictly in that order too.
  signal buf_issue_q : natural range 0 to g_assembly_buffers - 1 := 0;
  -- Buffer currently being presented on 'm_window_m2s'. Advances on every
  -- accepted beat; always the oldest buffer still holding a window.
  signal buf_out_q : natural range 0 to g_assembly_buffers - 1 := 0;
  -- Combinational: the buffer the *next* walk would be launched into.
  signal buf_next_c : natural range 0 to g_assembly_buffers - 1;

  -- Tap-assembly register: accumulates one full window's taps across the
  -- 'kc' walk, then is presented as 'm_window_m2s.data' once
  -- 'window_valid' is asserted. Cleared to all-'cfg_pad_value' at the
  -- start of every walk (ISA v2.1; it was all-zero before, which is what
  -- the default 'cfg_pad_value' of 0 still gives), so taps that are
  -- out-of-frame (padding) or beyond the runtime 'kh'/'kw' simply stay at
  -- the pad value without being written -- same structure as the
  -- predecessor combinational 'data_i', one fill value later.
  --
  -- Note this makes the *unused* taps (beyond 'kernel_h * kernel_w') read
  -- as the pad value too, not as 0. Every consumer masks by the runtime
  -- kernel size -- 'cnn_accel_pool' reduces only its 'active_count'
  -- lowest taps, 'cnn_accel_pe_array' only walks 'kernel_h * kernel_w' --
  -- so this is not observable; it is called out because a future consumer
  -- that reduced over the whole fixed array would silently break.
  --
  -- N-buffered ('g_assembly_buffers'). A buffer is *reserved* from the
  -- cycle its window's walk is launched until the cycle that window is
  -- accepted on 'm_window'; with 'kernel_w' columns issued one per cycle
  -- and a two-cycle read/capture pipeline behind them, that reservation
  -- lasts 'kernel_w + 2' cycles, so N buffers sustain one window every
  -- 'max(kernel_w, ceil((kernel_w + 2) / N))' cycles. At 'kernel_w = 1'
  -- that is 4 cycles for N = 1 (the pre-rework figure, counting the idle
  -- re-arm cycle this rework also removes), 1.5 for N = 2 and 1 for
  -- N = 3.
  subtype window_taps_t is tap_array_t(0 to c_window_data_length - 1);
  type assembly_arr_t is array (0 to g_assembly_buffers - 1) of window_taps_t;
  signal assembly_q : assembly_arr_t := (others => (others => (others => '0')));

  -- Per-buffer "holds a complete, not-yet-accepted window" flag. Set by
  -- the capture stage on the window's last column, cleared on acceptance.
  signal full_q : std_ulogic_vector(0 to g_assembly_buffers - 1) := (others => '0');

  -- Per-buffer stream sidebands, latched at walk launch. With more than
  -- one window in flight these can no longer be decoded from
  -- 'out_row_q'/'out_col_q'/'rd_tile_q' at presentation time -- those
  -- counters now belong to the walk being *launched*, which is up to
  -- 'g_assembly_buffers - 1' windows ahead of the one being presented.
  signal meta_first_tile_q : std_ulogic_vector(0 to g_assembly_buffers - 1) :=
    (others => '0');
  signal meta_last_tile_q : std_ulogic_vector(0 to g_assembly_buffers - 1) :=
    (others => '0');
  signal meta_last_q : std_ulogic_vector(0 to g_assembly_buffers - 1) :=
    (others => '0');

  -- Per-walk snapshots of the two 'control'-maintained geometry values the
  -- column walk consumes *while it runs*. They used to be read live from
  -- 'kr_base_q'/'row_ok_q', which was safe only because 'control' advanced
  -- 'out_row_q' on acceptance, i.e. never during the walk. Now 'control'
  -- advances on *launch*, so by the time a walk's second column is issued
  -- the live registers may already describe the next output row. Latching
  -- them at launch keeps the walk reading its own window's geometry, and
  -- costs one flip-flop stage on a path ('kr_base_walk_q' -> registered
  -- 'kr_base_capture_q') that was already register-to-register.
  signal kr_base_walk_q : kr_base_arr_t := (others => 0);
  signal row_ok_walk_q : flag_arr_t := (others => '0');

  signal fire : std_ulogic;
  -- '1' once the buffer currently at the head of the presentation order
  -- holds a fully assembled window. A 'g_assembly_buffers'-wide mux of
  -- registers, never a combinational function of any handshake input, so
  -- 'm_window_m2s.valid' still cannot depend on 'm_window_s2m.ready'.
  signal window_valid : std_ulogic;
  signal consume : std_ulogic;
  -- Stream sidebands of the *launching* window (they become that window's
  -- 'meta_*_q(buf)' entry), as opposed to the presented one.
  signal launch_last_pixel : std_ulogic;
  signal launch_first_tile, launch_last_tile : std_ulogic;
  -- '1' when a new column walk starts this cycle: the read side is free,
  -- a buffer is (or is being) freed, and this window's inputs are all
  -- written. Everything it depends on is either a register or
  -- 'm_window_s2m.ready' -- never 'window_valid', so no loop.
  signal walk_start : std_ulogic;
  -- '1' while the read side is free to take a new walk: no walk in
  -- progress, or the one in progress is issuing its final column.
  signal issue_free : std_ulogic;
  -- '1' while fewer than 'g_assembly_buffers' windows are reserved, or
  -- one is being freed this very cycle.
  signal buffer_free : std_ulogic;
  -- Windows launched but not yet accepted, i.e. buffers reserved.
  signal n_res_q : natural range 0 to g_assembly_buffers := 0;
  -- Combinational: '1' once every real (unpadded) row/column this window
  -- needs has been fully written -- unchanged formula from the
  -- predecessor design (cnn_accel_window_gen_proposal.md section 4), just
  -- no longer wired directly to 'window_valid'. Only ever sampled from
  -- the 'idle' state below (registered state only, so no combinational
  -- loop through 's_stream_m2s.valid'/'m_window_s2m.ready').
  signal row_ready_i : std_ulogic;

  -- M7 correctness fix, N-buffer generalization: combinational, '1' once
  -- the write side is about to advance into a physical row that would
  -- alias (same 'mod g_max_kernel_size' bank) the earliest real row that
  -- the OLDEST window still in flight needs. Gates 's_stream_s2m.ready'
  -- below, and since the M7 rework it is the ONLY thing that protects a
  -- row bank -- see the 's_stream_s2m.ready' comment for the full
  -- invariant and why the predecessor's additional "freeze while a window
  -- is pending" term is both unnecessary and, with several windows in
  -- flight, fatal to throughput.
  signal write_freeze_i : std_ulogic;

  -- The anchor 'write_freeze_i' compares against: the reservation queue's
  -- head when any window is in flight, the next-to-launch position's own
  -- anchor otherwise. Combinational, over registers only.
  signal anchor_head_c : coord_t;

  -- '1' once a frame is in progress (between 'start' and the final
  -- window's acceptance); gates 's_stream_s2m.ready' so nothing is
  -- accepted before the first 'start' or after the frame's last beat.
  -- High for the single cycle between 'start' and 'active_q': frame setup
  -- stage 2 (see the 'control' process). Nothing outside this process
  -- observes it -- the world sees only that 'active_q' rises one cycle
  -- later than it used to.
  signal setup_q : std_ulogic := '0';
  -- High for the cycle after 'setup_q': frame setup stage 3.
  signal setup2_q : std_ulogic := '0';

  signal active_q : std_ulogic := '0';
  -- '1' while windows remain to be *launched* (between 'start' and the
  -- launch of the final window). Distinct from 'active_q', which only
  -- drops 'kernel_w + 2' cycles later, when that final window is
  -- accepted.
  signal launch_active_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- Configuration, latched at 'start'.
  ------------------------------------------------------------------------

  signal kernel_h_q, kernel_w_q : unsigned(7 downto 0) := (others => '0');
  signal stride_h_q, stride_w_q : unsigned(7 downto 0) := (others => '0');
  signal pad_top_q, pad_left_q : unsigned(7 downto 0) := (others => '0');
  -- ISA v2.1 pad value, latched at 'start' alongside the pad counts.
  -- The int8 value every padded tap takes. It is written to every byte of
  -- every tap-assembly buffer at the start of a column walk, so its eight
  -- bits drive 'g_max_kernel_size**2 * g_tile_channels *
  -- g_assembly_buffers' flip-flop inputs -- 216 loads per bit for the CONV
  -- instance. Once the logic depth in this entity was gone, that single
  -- net was the accelerator's worst path: one LUT level, 0.5 ns of logic
  -- and 8.5 ns of route. 'max_fanout' makes the synthesiser replicate the
  -- register instead of driving one net across the whole buffer bank.
  signal pad_value_q : std_ulogic_vector(7 downto 0) := (others => '0');
  attribute max_fanout : integer;
  attribute max_fanout of pad_value_q : signal is 24;
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
  --     'kr_base_capture_q'/'in_frame_capture_q'/'capture_valid_q' one-deep
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

  -- Reservation queue of 'anchor_limit_q' snapshots, one per window in
  -- flight, oldest first: pushed at walk launch, popped on acceptance,
  -- at most 'g_assembly_buffers' deep by construction (that is exactly
  -- what 'n_res_q' counts). 'anchor_limit_q' itself is a property of the
  -- window being LAUNCHED, and with several windows in flight that is no
  -- longer the window whose row banks are most at risk -- the oldest one
  -- is. Since 'row_top' is non-decreasing in the output row, so is
  -- 'anchor_limit', which makes the head of this queue the minimum and
  -- therefore the only entry 'write_freeze_i' has to compare against.
  type anchor_arr_t is array (0 to g_assembly_buffers - 1) of coord_t;
  signal anchor_res_q : anchor_arr_t := (others => g_max_kernel_size);

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

  ------------------------------------------------------------------------
  -- Registered pad-clear of the reserved assembly buffer.
  --
  -- This is a pure timing rework of the clear that used to sit inside the
  -- 'walk_start' branch of 'walk_control'; the values written are
  -- identical, only the cycle they are written in moves by one.
  --
  -- Why it had to move (shared/TimingAndResources.md, "Fan-out is a
  -- timing path", plus "Handshake stages"). 'walk_start' is
  -- combinational: 'launch_active_q and row_ready_i and issue_free and
  -- buffer_free', and 'buffer_free' contains 'consume', which contains
  -- 'm_window_s2m.ready' -- the CONSUMER's ready. Driving the clear from
  -- it therefore put the consumer's ready, and everything upstream of it,
  -- on the clock-enable pin of all 'g_assembly_buffers x
  -- g_max_kernel_size**2 x g_tile_channels' assembly flip-flops (1600 per
  -- buffer at the conv geometry, 1600 at the pool one). In the routed
  -- top-level build that single cone owned 278 of the 400 worst paths:
  --
  --   'elementwise/state_q_reg[4] -> cmd_proc -> pe_array ->
  --    window_gen/consume -> conv window_gen assembly_q_reg[*][*][*]/CE'
  --   -1.883 ns, 12 logic levels, 6.6 ns of it route
  --
  -- and the pool instance had the same shape from its lanes' 'cfg_match'
  -- ('pool_inst/cfg_kernel_w_q -> pool_window_gen/assembly_q_reg/CE',
  -- -1.900 ns, 10 levels).
  --
  -- Delaying the clear by one cycle makes the enable of those ~5000
  -- flip-flops a plain register output. It is free, because the clear has
  -- a cycle of slack by construction: a walk armed in cycle t issues its
  -- 'kc = 0' address in t+1 and its first capture lands at the end of
  -- t+2, so a clear taking effect at the end of t+1 is still strictly
  -- before anything is written into the buffer.
  --
  -- It cannot collide with the PREVIOUS walk's final capture either, at
  -- any 'g_assembly_buffers':
  --   * N >= 2: the capture at t+1 belongs to 'buf_issue_q' as it was at
  --     t, and 'buf_next_c = next_buf(buf_issue_q)' is a different buffer.
  --   * N = 1: 'buffer_free' is 'n_res_q < 1 or consume', so a walk can
  --     only be armed in a cycle where the previous window is being
  --     accepted -- which cannot happen before that window is 'full_q',
  --     i.e. before its last capture has already completed. The
  --     'issue_free' fast path ("arm while the previous walk issues its
  --     last column") is therefore unreachable at N = 1, and no capture
  --     is in flight at t+1 at all.
  ------------------------------------------------------------------------
  signal clear_q : std_ulogic := '0';
  signal clear_buf_q : natural range 0 to g_assembly_buffers - 1 := 0;

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

  -- Next index in the 'g_assembly_buffers'-deep round-robin buffer ring.
  --
  -- Deliberately written through a WIDENED intermediate instead of the
  -- obvious '0 when idx = g_assembly_buffers - 1 else idx + 1'. At
  -- 'g_assembly_buffers = 1' -- the POOL instance -- the buffer-index
  -- subtype is '0 to 0', and GHDL's synthesis frontend statically
  -- range-checks BOTH arms of a conditional, including the one the guard
  -- makes unreachable, so 'idx + 1' is rejected outright with "value out
  -- of range". Simulation is perfectly happy with the same code (the else
  -- arm is never selected), so this is invisible to the whole testbench
  -- suite at either buffer depth -- it was caught by this module's own
  -- netlist build at the pool geometry, and only there. Same class of
  -- GHDL-synthesis strictness as the variable-loop-bound trap documented
  -- in 'control's D11 comment, and the reason 'v' is bounded at
  -- '2 * g_assembly_buffers' is simply that it must hold the intermediate
  -- 'idx + 1' for every legal 'idx', N = 1 included.
  function next_buf(idx : natural) return natural is
    variable v : natural range 0 to 2 * g_assembly_buffers;
  begin
    v := idx + 1;
    if v >= g_assembly_buffers then
      return 0;
    end if;
    return v;
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

  ------------------------------------------------------------------------
  -- THE ROW-BANK INVARIANT.
  --
  -- What has to be true: a row bank must not be overwritten while any
  -- window that reads from it is still being assembled or presented.
  --
  -- The rule this module used to enforce was a conjunction of two:
  --
  --   (a) "freeze input acceptance whenever a window is pending" --
  --       'not (window_valid or reading_q)', relaxed only on the cycle
  --       the pending window is accepted; and
  --   (b) 'write_freeze_i': freeze once 'cur_row_q' reaches
  --       'anchor_limit = max(row_top, 0) + g_max_kernel_size', i.e. the
  --       first physical input row whose bank number
  --       ('row mod g_max_kernel_size') aliases the earliest real row the
  --       current output row's windows still need.
  --
  -- (a) is the older, coarse rule; (b) was added by M7 precisely because
  -- (a) turned out NOT to be sufficient (a slow 1x1 consumer let the
  -- writer outrun the reader by a full 'g_max_kernel_size' rows between
  -- pending-window pulses). The re-derivation this rework needed is the
  -- other direction: given (b), is (a) needed at all? It is not, and it
  -- had to go, because (a) caps input acceptance at roughly one beat per
  -- window beat *turnaround* -- with the walk itself now able to retire
  -- one window per cycle, (a) would simply starve the row banks instead.
  --
  -- Why (b) alone is sufficient. Within one physical input row the write
  -- side only ever moves forward (one cell per accepted beat, restarting
  -- at cell 0 on a row boundary -- see 'control'), so it never rewrites a
  -- cell of a row it is already inside. The ONLY way a cell that some
  -- window still needs can be rewritten is therefore for the write side
  -- to advance into a *later* physical row that lands in the same bank,
  -- i.e. one exactly 'g_max_kernel_size' rows on from a row still in use.
  -- Let 'R = max(row_top, 0)' be the earliest real row that window needs
  -- ('row_top' clamped up because a negative 'row_top' is top padding,
  -- which reserves no bank). Rows 'R .. R + g_max_kernel_size - 1'
  -- occupy 'g_max_kernel_size' distinct banks, and 'kernel_h <=
  -- g_max_kernel_size' (asserted at 'start') puts every row that window
  -- reads inside that span. So no bank the window reads is aliased until
  -- the writer reaches row 'R + g_max_kernel_size' = 'anchor_limit', and
  -- (b) stops it there. Separately, 'row_ready_i' already guarantees that
  -- every cell the window reads was written *before* its walk started, so
  -- the walk never races an in-progress write to the same cell either.
  --
  -- What changes with N windows in flight. 'anchor_limit_q' tracks the
  -- output position 'control' is about to LAUNCH, which since this rework
  -- runs ahead of the position being presented. Applied to that position
  -- the rule is wrong: the halo case makes it concrete. 3x3, stride 1,
  -- two windows in flight, the older from output row 'r' (needing input
  -- rows 'r-1 .. r+1') and the newer from output row 'r+1' (needing
  -- 'r .. r+2') -- consecutive windows share two of their three rows, and
  -- the newer one's anchor, 'r + 1 + K', would let the writer into row
  -- 'r + K', whose bank is that of row 'r'... which is fine, but it would
  -- equally let it into row 'r + K - 1', whose bank is that of row
  -- 'r - 1' -- a row the OLDER window is still assembling from. The
  -- correct anchor is therefore the oldest in-flight window's, not the
  -- launching one's, which is what 'anchor_res_q' (see its declaration)
  -- keeps and 'anchor_head_c' selects. Because 'row_top' is
  -- non-decreasing in the output row, the oldest entry is also the
  -- smallest, so the head alone is the binding constraint and no minimum
  -- has to be computed.
  --
  -- Two smaller points, both deliberate:
  --   * a window is popped from 'anchor_res_q' at *acceptance*, not at
  --     the end of its walk. That is conservative (its banks are free one
  --     or two cycles earlier) and costs nothing: the writer is bank-
  --     limited, not cycle-limited.
  --   * 'write_freeze_i' no longer carries the predecessor's
  --     'has_real_row_q = 1' qualifier. That qualifier disabled the
  --     interlock entirely for a window with no real row at all, which is
  --     harmless for bottom padding (there 'anchor_limit' is already past
  --     the end of the frame and never binds) but NOT for a window lying
  --     entirely in top padding ('kernel_h <= pad_top'), where it let the
  --     writer run unbounded ahead of rows later windows do need.
  --     Dropping it is strictly tighter and cannot deadlock: whenever the
  --     freeze bites, 'cur_row_q >= anchor_limit > real_row_bot', so
  --     'row_ready_i' is already '1' and the reader can always make
  --     progress and move the anchor on.
  --
  -- No combinational loop: every term is a register except
  -- 'm_window_s2m.ready', which appears nowhere in this expression.
  ------------------------------------------------------------------------
  s_stream_s2m.ready <= active_q and not write_freeze_i;

  ------------------------------------------------------------------------
  -- Configuration latch, position counters, write-side address counters,
  -- and the S7 per-output-position geometry registers (see their
  -- declaration block above -- they are updated here, in the very same
  -- clocked branches that advance 'out_row_q'/'out_col_q'/'rd_tile_q',
  -- which is what keeps them exactly coherent with those counters rather
  -- than a cycle behind them).
  ------------------------------------------------------------------------
  control : process(clk)
    -- Ranged, not plain 'integer'. Every one of these is a copy of an
    -- 8- or 16-bit port or register, but as unconstrained integers they
    -- made every 'mod' and every product below 32 bits wide in the
    -- netlist: '(-pad_top) mod g_max_kernel_size' alone synthesised to a
    -- 32-bit signed modulo, 11.9 ns of it, and was the second-worst path
    -- in the design once the divisions were gone.
    variable v_in_channels : natural range 0 to 65535;
    variable v_n_tiles : natural range 0 to 65535;
    variable v_kh, v_kw, v_sh, v_sw, v_pt, v_pl : natural range 0 to 255;
    variable v_inh, v_inw : natural range 0 to 65535;
    variable v_row_top, v_col_left : coord_t;
    variable v_pt_mod : natural range 0 to g_max_kernel_size - 1;
    variable v_mod : natural range 0 to 2 * g_max_kernel_size - 2;
    variable v_kr : natural range 0 to 2 * g_max_kernel_size - 1;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        active_q <= '0';
        launch_active_q <= '0';
        setup_q <= '0';
        setup2_q <= '0';
        n_res_q <= 0;
        cur_row_q <= (others => '0');
        cur_col_q <= (others => '0');
        wr_tile_q <= (others => '0');
        out_row_q <= (others => '0');
        out_col_q <= (others => '0');
        rd_tile_q <= (others => '0');
        wr_bank_q <= 0;
        wr_addr_q <= 0;

      elsif start = '1' then
        -- ------------------------------------------------------------
        -- Frame setup, stage 1 of 2: latch the configuration and clear
        -- the position counters. 'active_q' stays low: the seeding of the
        -- geometry registers below happens in stage 2, off the registers
        -- latched here rather than off the 'cfg_*' ports.
        --
        -- Splitting what used to be one cycle in two is what takes
        -- 'cnn_accel_cmd_proc's descriptor register out of the seeding
        -- cone. Every product and modulo in stage 2 now starts at a
        -- register inside this entity, one command-start cycle later.
        -- The cost is exactly one cycle per COMMAND (commands run for
        -- thousands); nothing per position and nothing per beat, since
        -- 's_stream_s2m.ready' is gated by 'active_q' and simply
        -- backpressures the feeder for that cycle.
        -- ------------------------------------------------------------
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
        pad_value_q <= cfg_pad_value;
        in_width_q <= unsigned(cfg_in_width);
        in_height_q <= unsigned(cfg_in_height);

        -- 'out_dim = (in_dim + pad_lo + pad_hi - kernel) / stride + 1',
        -- received pre-divided from 'cnn_accel_cmd_proc' -- see the
        -- 'cfg_out_width' port comment for why it is not derived here.
        out_width_q <= unsigned(cfg_out_width);
        out_height_q <= unsigned(cfg_out_height);

        -- T = ceil(in_channels / g_tile_channels); only the last tile of
        -- the frame can be partial (see 'last_tile_channels_q's comment).
        -- 'g_tile_channels' is a generic, so this is a constant divide.
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
        active_q <= '0';
        launch_active_q <= '0';
        n_res_q <= 0;

        wr_bank_q <= 0;
        wr_addr_q <= 0;

        setup_q <= '1';
        setup2_q <= '0';

      elsif setup_q = '1' then
        -- ------------------------------------------------------------
        -- Frame setup, stage 2 of 2: seed every geometry register and
        -- address accumulator for 'out_row = out_col = rd_tile = 0', then
        -- go active.
        --
        -- Every source here is a register latched in stage 1, so the
        -- deepest cone in this block is one narrow multiply
        -- register-to-register, instead of the descriptor-register-to-here
        -- cone it used to be.
        -- ------------------------------------------------------------
        setup_q <= '0';

        v_kh := to_integer(kernel_h_q);
        v_kw := to_integer(kernel_w_q);
        v_sh := to_integer(stride_h_q);
        v_sw := to_integer(stride_w_q);
        v_pt := to_integer(pad_top_q);
        v_pl := to_integer(pad_left_q);
        v_inh := to_integer(in_height_q);
        v_inw := to_integer(in_width_q);
        v_n_tiles := to_integer(n_tiles_q);

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

        -- The two modulo-'g_max_kernel_size' reductions are the deepest
        -- single operation in the setup, and 'row_top_mod_q' feeds a
        -- second layer of per-bank selects, multiplies and range tests.
        -- Both are registered here and consumed in stage 3, so no cone
        -- runs 'pad_top -> mod -> kr -> row_ok' in one cycle.
        stride_h_mod_q <= v_sh mod g_max_kernel_size;

        -- '(-pad_top) mod K', written as a reduction of the NON-negative
        -- 'pad_top mod K'. Identical by definition -- VHDL's 'mod' takes
        -- the sign of its right operand, so '(-p) mod K' is the unique
        -- 'r' in '0 .. K-1' congruent to '-p', which is 'K - (p mod K)'
        -- unless 'p mod K' is zero. The point is that the operand is now
        -- an 8-bit non-negative value instead of a signed integer.
        v_pt_mod := v_pt mod g_max_kernel_size;
        if v_pt_mod = 0 then
          row_top_mod_q <= 0;
        else
          row_top_mod_q <= g_max_kernel_size - v_pt_mod;
        end if;

        -- 'k * kernel_w' for every possible kernel row, so the
        -- per-output-row 'kr_base_q' update below is a mux, not a
        -- multiply.
        for k in 0 to g_max_kernel_size - 1 loop
          kr_kw_q(k) <= k * v_kw;
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

        setup2_q <= '1';

      elsif setup2_q = '1' then
        -- ------------------------------------------------------------
        -- Frame setup, stage 3 of 3: the per-row-bank kernel-row mapping,
        -- which is everything downstream of 'row_top_mod_q'. Kept out of
        -- stage 2 because the chain 'pad_top -> mod g_max_kernel_size ->
        -- kernel row -> (multiply, range tests) -> row_ok_q' was still a
        -- 9 ns cone at 150 MHz when it ran in one cycle.
        --
        -- Reads only registers written in stages 1 and 2. 'active_q'
        -- rises at the end of this cycle, so the frame is live from here.
        -- ------------------------------------------------------------
        setup2_q <= '0';
        active_q <= '1';
        launch_active_q <= '1';

        v_kh := to_integer(kernel_h_q);
        v_kw := to_integer(kernel_w_q);
        v_pt := to_integer(pad_top_q);
        v_inh := to_integer(in_height_q);
        v_row_top := -v_pt;

        for b in 0 to g_max_kernel_size - 1 loop
          v_kr := b + g_max_kernel_size - row_top_mod_q;
          if v_kr >= g_max_kernel_size then
            v_kr := v_kr - g_max_kernel_size;
          end if;
          kr_of_q(b) <= v_kr;
          -- Start-time only: one narrow multiply per bank, off every
          -- cycle-by-cycle path (see 'kr_base_q's declaration).
          kr_base_q(b) <= v_kr * v_kw;
          row_ok_q(b) <= to_sl(
            v_kr < v_kh and v_row_top + v_kr >= 0 and v_row_top + v_kr <= v_inh - 1
          );
        end loop;

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

        -- Reservation queue of row-bank anchors, one entry per window in
        -- flight (see 'anchor_res_q' / the 's_stream_s2m.ready' comment).
        -- Pushed with the LAUNCHING window's anchor, popped on
        -- acceptance; 'walk_start' can never fire when the queue is full
        -- unless 'consume' frees a slot the same cycle ('buffer_free'),
        -- so 'n_res_q' stays in range by construction.
        if walk_start = '1' and consume = '1' then
          if n_res_q = g_assembly_buffers then
            for i in 0 to g_assembly_buffers - 2 loop
              anchor_res_q(i) <= anchor_res_q(i + 1);
            end loop;
            anchor_res_q(g_assembly_buffers - 1) <= anchor_limit_q;
          else
            for i in 0 to g_assembly_buffers - 1 loop
              if i = n_res_q - 1 then
                anchor_res_q(i) <= anchor_limit_q;
              elsif i < g_assembly_buffers - 1 then
                anchor_res_q(i) <= anchor_res_q(i + 1);
              end if;
            end loop;
          end if;
        elsif walk_start = '1' then
          for i in 0 to g_assembly_buffers - 1 loop
            if i = n_res_q then
              anchor_res_q(i) <= anchor_limit_q;
            end if;
          end loop;
          n_res_q <= n_res_q + 1;
        elsif consume = '1' then
          for i in 0 to g_assembly_buffers - 2 loop
            anchor_res_q(i) <= anchor_res_q(i + 1);
          end loop;
          n_res_q <= n_res_q - 1;
        end if;

        -- The frame is over, for input-acceptance purposes, once its
        -- final window has been ACCEPTED (unchanged from the predecessor
        -- design, where the same test lived in the launch branch below
        -- because launch and acceptance were the same event).
        if consume = '1' and meta_last_q(buf_out_q) = '1' then
          active_q <= '0';
        end if;

        -- Output-position counters and the whole S7 geometry block now
        -- advance on walk LAUNCH, not on acceptance: they define the
        -- addresses the read side is about to issue, and with N windows
        -- in flight the read side runs ahead of the presentation side.
        -- Everything that must stay with the *presented* window (the
        -- stream sidebands) is snapshotted into 'meta_*_q' by
        -- 'walk_control' in this same cycle; everything the walk itself
        -- consumes ('kr_base_q'/'row_ok_q') into 'kr_base_walk_q'/
        -- 'row_ok_walk_q'.
        if walk_start = '1' then
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
                  kr_base_q(b) <= kr_kw_q(v_kr);
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

            if launch_last_pixel = '1' then
              launch_active_q <= '0';
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
  -- Sidebands of the window being LAUNCHED this cycle; 'walk_control'
  -- latches them into that window's buffer, and the presentation side
  -- below reads them back out of 'meta_*_q'.
  launch_last_pixel <= '1'
    when (out_row_q = out_height_q - 1 and out_col_q = out_width_q - 1
          and rd_tile_q = n_tiles_q - 1)
    else '0';
  launch_first_tile <= '1' when rd_tile_q = 0 else '0';
  launch_last_tile <= '1' when rd_tile_q = n_tiles_q - 1 else '0';

  ------------------------------------------------------------------------
  -- Presentation side: an N:1 mux of registers, selected by the
  -- oldest-buffer pointer. Nothing here is a function of any handshake
  -- input, so 'm_window_m2s.valid' still depends only on registered
  -- state -- which is what lets 'cnn_accel_conv_core' wire this straight
  -- to 'cnn_accel_pe_array' with no skid register (see that entity's
  -- header comment).
  --
  -- NOTE: explicit sensitivity list, not 'process(all)' -- GHDL
  -- 7.0.0-dev's 'all' inference does not reliably track signals read only
  -- through nested loops/array indexing.
  ------------------------------------------------------------------------
  present : process(buf_out_q, full_q, assembly_q, meta_first_tile_q, meta_last_tile_q, meta_last_q)
  begin
    window_valid <= '0';
    m_window_m2s.data <= (others => (others => '0'));
    m_window_m2s.last <= '0';
    m_window_m2s.first_tile <= '0';
    m_window_m2s.last_tile <= '0';
    for buf in 0 to g_assembly_buffers - 1 loop
      if buf_out_q = buf then
        window_valid <= full_q(buf);
        m_window_m2s.data <= assembly_q(buf);
        m_window_m2s.last <= meta_last_q(buf);
        m_window_m2s.first_tile <= meta_first_tile_q(buf);
        m_window_m2s.last_tile <= meta_last_tile_q(buf);
      end if;
    end loop;
  end process;

  m_window_m2s.valid <= window_valid;
  done <= consume and m_window_m2s.last;

  ------------------------------------------------------------------------
  -- Walk-launch arbitration (combinational, over registers plus
  -- 'm_window_s2m.ready'). A walk may start when
  --   * the read side is free ('issue_free': no walk, or the walk in
  --     progress is issuing its final column, so the next walk's first
  --     address can be loaded this very cycle -- this is where the
  --     predecessor's idle re-arm cycle went),
  --   * a buffer is free or is being freed this cycle ('buffer_free'),
  --     and
  --   * this window's inputs are all written ('row_ready_i'), and there
  --     are windows left to launch ('launch_active_q').
  -- 'buffer_free' is what keeps the row-bank/backpressure argument
  -- intact: a buffer is reserved from launch to acceptance, so a stalled
  -- consumer stops launches after at most 'g_assembly_buffers' windows
  -- and nothing in flight is ever overwritten or reordered.
  ------------------------------------------------------------------------
  issue_free <= '1' when issue_q = '0' or kc_q = kernel_w_q - 1 else '0';
  buffer_free <= '1' when n_res_q < g_assembly_buffers or consume = '1' else '0';
  walk_start <= launch_active_q and row_ready_i and issue_free and buffer_free;

  buf_next_c <= next_buf(buf_issue_q);

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
  anchor_head_c <= anchor_limit_q when n_res_q = 0 else anchor_res_q(0);

  read_qualify : process(
    cur_row_q, cur_col_q, issue_q, rd_col_ok_q,
    has_real_row_q, real_row_bot_q, has_real_col_q, real_col_right_q,
    anchor_head_c, row_ok_walk_q
  )
    variable v_cur_row : coord_t;
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

    if v_cur_row >= anchor_head_c then
      write_freeze_i <= '1';
    else
      write_freeze_i <= '0';
    end if;

    for b in 0 to g_max_kernel_size - 1 loop
      in_frame_now(b) <= issue_q and row_ok_walk_q(b) and rd_col_ok_q;
    end loop;
  end process;

  ------------------------------------------------------------------------
  -- Column walk (read side), tap-assembly capture, and the N-deep
  -- assembly-buffer pipeline. Four independent stages, all of which
  -- advance every cycle -- there is no longer a walk "state machine" with
  -- an idle state at all:
  --
  -- 1. Issue: while 'issue_q', 'rd_addr_q' presents one column address
  --    per cycle for 'kc = 0 .. kernel_w - 1'. Unlike the predecessor
  --    there is no 'kc = kernel_w' drain step and no idle re-arm cycle:
  --    a new walk is armed in the same cycle the previous one issues its
  --    last column, so back-to-back windows issue addresses with no gap
  --    and the issue side alone is the throughput bound at 'kernel_w'
  --    cycles per window.
  -- 2. Capture (one cycle later, the row banks' registered read
  --    latency): pack 'bank_rd_data' into 'assembly_q(buf_capture_q)' at
  --    tap slot 'kr_base_capture_q(b) + kc_capture_q' (= 'kr * kernel_w
  --    + kc', with the multiply hoisted onto the output-row boundary
  --    path -- see 'kr_base_q's declaration for the measured timing
  --    reason), for whichever banks were in-frame. 'tap_idx' depends on
  --    the runtime 'kw', so it cannot index 'assembly_q' directly
  --    without becoming a decoder anyway; the destination slot is
  --    selected with a constant-bound loop, identical in simulation, a
  --    mux in hardware -- the same technique the predecessor's
  --    combinational process used for the same reason.
  -- 3. Completion: the capture of a window's last column ('capture_last_q')
  --    marks its buffer 'full_q'. That IS the old drain cycle, now doing
  --    useful work for the *next* window instead of stalling this one.
  -- 4. Presentation: the 'present' process above muxes the oldest full
  --    buffer onto 'm_window_m2s'; acceptance clears its 'full_q' and
  --    advances 'buf_out_q'.
  --
  -- End to end a window takes 'kernel_w + 2' cycles from launch to
  -- acceptance (arm, 'kernel_w' issue cycles, one capture-latency cycle,
  -- with the last capture and the first present overlapping), and a
  -- buffer is reserved for exactly that span. With 'g_assembly_buffers'
  -- of them the sustained rate is therefore
  -- 'max(kernel_w, ceil((kernel_w + 2) / N))' cycles per window --
  -- 'kernel_w'-bound (i.e. optimal) for every 'kernel_w >= 2' at N = 2,
  -- and 1 cycle per window at 'kernel_w = 1' only from N = 3 up, which is
  -- why the CONV instance uses 3.
  --
  -- Ordering and backpressure. Buffers are allocated strictly in launch
  -- order ('buf_issue_q' advancing mod N) and released strictly in that
  -- same order ('buf_out_q'), so windows can neither be reordered nor
  -- merged. A stalled consumer stops 'consume', which stops 'buf_out_q'
  -- and 'n_res_q' from draining, which clears 'buffer_free' and therefore
  -- 'walk_start' -- launches halt after at most N windows are in flight
  -- and every one of them is held intact in its own buffer until the
  -- consumer returns. Nothing is dropped, because a walk is never started
  -- for a buffer that is not free, and nothing is overwritten, because a
  -- buffer's 'assembly_q' is only ever written by the walk that reserved
  -- it.
  --
  -- 'row_ready_i' still gates every launch, i.e. a walk only starts once
  -- every input pixel that window needs has been written -- unchanged
  -- invariant from the predecessor design (proposal doc section 7.1
  -- item 4 / section 8 risk 3). The row banks those pixels live in are
  -- protected for the whole time the window is in flight by
  -- 'write_freeze_i' against 'anchor_head_c'; see the 's_stream_s2m.ready'
  -- comment for that derivation.
  --
  -- S7 read-address accumulators (unchanged): 'rd_word_q'/'rd_col_q' are
  -- loaded -- from registers 'control' keeps coherent with
  -- 'out_col_q'/'rd_tile_q' -- in the *same* cycle the walk is armed, so
  -- the 'kc = 0' address is presented the very next cycle. Each
  -- subsequent 'kc' step is '+ n_tiles_words_q' (one input column =
  -- 'n_tiles_q' consecutive channel-tile cells), replacing the
  -- predecessor's 'input_col * n_tiles_q + rd_tile_q' multiply.
  -- 'rd_addr_q' is that accumulator clamped to a legal address, so the
  -- BRAM address port is driven by a flip-flop and nothing else; the
  -- clamp condition is the same '[0, in_width_q - 1]' test that gates
  -- 'in_frame_now', so an out-of-frame cycle reads cell 0 exactly as the
  -- predecessor's 'rd_addr(b) <= 0' branch did.
  ------------------------------------------------------------------------
  walk_control : process(clk)
    variable v_next_word : word_t;
    variable v_next_col : coord_t;
    variable v_next_ok : boolean;
  begin
    if rising_edge(clk) then
      if reset = '1' or start = '1' then
        issue_q <= '0';
        kc_q <= (others => '0');
        capture_valid_q <= '0';
        capture_last_q <= '0';
        kc_capture_q <= (others => '0');
        in_frame_capture_q <= (others => '0');
        kr_base_capture_q <= (others => 0);
        tap_sel_capture_q <= (others => (others => '0'));
        kr_base_walk_q <= (others => 0);
        row_ok_walk_q <= (others => '0');
        assembly_q <= (others => (others => (others => '0')));
        full_q <= (others => '0');
        meta_first_tile_q <= (others => '0');
        meta_last_tile_q <= (others => '0');
        meta_last_q <= (others => '0');
        -- Seeded one short of 0 so the first walk of a frame lands in
        -- buffer 0, which is where 'buf_out_q' starts: launch order and
        -- presentation order have to agree from the first window.
        buf_issue_q <= g_assembly_buffers - 1;
        buf_capture_q <= 0;
        buf_out_q <= 0;
        rd_addr_q <= 0;
        rd_word_q <= (others => '0');
        rd_col_q <= 0;
        rd_col_ok_q <= '0';
        clear_q <= '0';
        clear_buf_q <= 0;

      else
        --------------------------------------------------------------
        -- 1. Delay pipeline: always shifts by one cycle, so the capture
        --    stage below sees last cycle's issued column/in-frame/bank
        --    mapping (and its destination buffer) alongside this cycle's
        --    now-valid 'bank_rd_data'.
        --------------------------------------------------------------
        capture_valid_q <= issue_q;
        capture_last_q <= issue_q and to_sl(kc_q = kernel_w_q - 1);
        -- Registered pad-clear request, see the 'clear_q' declaration.
        clear_q <= walk_start;
        clear_buf_q <= buf_next_c;
        kc_capture_q <= kc_q;
        kr_base_capture_q <= kr_base_walk_q;
        in_frame_capture_q <= in_frame_now;
        buf_capture_q <= buf_issue_q;
        -- Same tap slot, one-hot -- see 'tap_sel_capture_q'. Both loop
        -- bounds are generics, so this is a fixed comparator array, not a
        -- variable-bound loop.
        for b in 0 to g_max_kernel_size - 1 loop
          for t in 0 to g_max_kernel_size * g_max_kernel_size - 1 loop
            tap_sel_capture_q(b)(t) <=
              to_sl(t = kr_base_walk_q(b) + to_integer(kc_q));
          end loop;
        end loop;

        --------------------------------------------------------------
        -- 1b. Registered pad-clear of the buffer reserved one cycle ago.
        --     Written BEFORE the capture stage below so that, if a future
        --     change ever did make the two collide on one buffer, the
        --     capture would win and a real tap could not be erased.
        --------------------------------------------------------------
        if clear_q = '1' then
          for buf in 0 to g_assembly_buffers - 1 loop
            if clear_buf_q = buf then
              assembly_q(buf) <= (others => pad_value_q);
            end if;
          end loop;
        end if;

        --------------------------------------------------------------
        -- 2. Capture stage. Same tap decode as before ('kr * kernel_w +
        --    kc', with the multiply hoisted onto the output-row boundary
        --    path -- see 'kr_base_q'), now with the destination buffer
        --    selected by an equality against a one-hot-decoded
        --    'buf_capture_q' rather than a direct runtime index, for the
        --    same reason every other runtime-indexed write in this file
        --    is written that way.
        --------------------------------------------------------------
        if capture_valid_q = '1' then
          for buf in 0 to g_assembly_buffers - 1 loop
            if buf_capture_q = buf then
              for b in 0 to g_max_kernel_size - 1 loop
                if in_frame_capture_q(b) = '1' then
                  for t in 0 to g_max_kernel_size * g_max_kernel_size - 1 loop
                    if tap_sel_capture_q(b)(t) = '1' then
                      for c in 0 to g_tile_channels - 1 loop
                        assembly_q(buf)(t * g_tile_channels + c) <=
                          bank_rd_data(b)(8 * (c + 1) - 1 downto 8 * c);
                      end loop;
                    end if;
                  end loop;
                end if;
              end loop;
            end if;
          end loop;
        end if;

        --------------------------------------------------------------
        -- 3. Presentation side. A buffer is freed by acceptance and
        --    filled by its window's final capture; those can never be
        --    the same buffer on the same cycle (a buffer being written
        --    is by definition not full, and only a full buffer can be
        --    accepted), so the order of these two branches is immaterial
        --    -- but the fill is written second anyway, so that a future
        --    change which does make them collide fails safe (full wins)
        --    rather than silently dropping a window.
        --------------------------------------------------------------
        if consume = '1' then
          for buf in 0 to g_assembly_buffers - 1 loop
            if buf_out_q = buf then
              full_q(buf) <= '0';
            end if;
          end loop;
          buf_out_q <= next_buf(buf_out_q);
        end if;

        if capture_last_q = '1' then
          for buf in 0 to g_assembly_buffers - 1 loop
            if buf_capture_q = buf then
              full_q(buf) <= '1';
            end if;
          end loop;
        end if;

        --------------------------------------------------------------
        -- 4. Issue side. One column address per cycle, 'kc = 0 ..
        --    kernel_w - 1'; a new walk is armed in the same cycle the
        --    previous one issues its last column, so back-to-back
        --    windows leave no bubble at all. 'walk_start' (see its
        --    combinational definition above) is exactly "the read side
        --    is free AND a buffer is free AND this window's inputs are
        --    written", so the two branches below are mutually exclusive
        --    by construction: 'walk_start' can only be '1' when
        --    'issue_free' is, i.e. when the 'elsif' would have been
        --    taking the 'kc_q = kernel_w_q - 1' exit anyway.
        --
        --    'rd_word_q'/'rd_col_q' are loaded from registers 'control'
        --    keeps coherent with 'out_col_q'/'rd_tile_q' in the same
        --    cycle 'control' advances them, so the 'kc = 0' address is
        --    presented on the very next cycle. 'kr_base_walk_q'/
        --    'row_ok_walk_q' are snapshotted here for the same reason
        --    (see their declaration): 'control' has already moved on.
        --------------------------------------------------------------
        if walk_start = '1' then
          issue_q <= '1';
          kc_q <= (others => '0');
          buf_issue_q <= buf_next_c;
          kr_base_walk_q <= kr_base_q;
          row_ok_walk_q <= row_ok_q;

          for buf in 0 to g_assembly_buffers - 1 loop
            if buf_next_c = buf then
              -- Only the (three-bit, low-fanout) per-window metadata is
              -- written here. The buffer's all-'cfg_pad_value' clear --
              -- which is what makes a tap left out-of-frame this window
              -- (padding, or beyond the runtime 'kh'/'kw') read back as
              -- padding without being written -- is issued one cycle
              -- later from the registered 'clear_q'/'clear_buf_q'; see
              -- their declaration for why, and for why the delay is
              -- safe at every 'g_assembly_buffers'.
              meta_first_tile_q(buf) <= launch_first_tile;
              meta_last_tile_q(buf) <= launch_last_tile;
              meta_last_q(buf) <= launch_last_pixel;
            end if;
          end loop;

          v_next_ok := col_left_q >= 0 and col_left_q <= in_width_m1_q;
          rd_word_q <= rd_base_q;
          rd_col_q <= col_left_q;
          rd_col_ok_q <= to_sl(v_next_ok);
          if v_next_ok then
            rd_addr_q <= to_integer(rd_base_q);
          else
            rd_addr_q <= 0;
          end if;

        elsif issue_q = '1' then
          if kc_q = kernel_w_q - 1 then
            issue_q <= '0';
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
        end if;
      end if;
    end if;
  end process;

end architecture a;
