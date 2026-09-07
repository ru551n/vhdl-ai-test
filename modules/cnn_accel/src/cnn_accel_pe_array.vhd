library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library math;
use math.math_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- int8 x int8 multiply-accumulate array with input-channel-tile partial-sum
-- carry ("M4", the tiled dataflow rewrite). See
-- modules/cnn_accel/doc/cnn_accel_pe_array_req.md,
-- doc/cnn_accel_pe_array_proposal.md section 3.5/3.6/6 (the broadcast-
-- activation/per-lane-weight compute shape and one-entry-output-register
-- idiom, unchanged by tiling) and doc/cnn_accel_tiled_dataflow_proposal.md
-- section 3/4/7 (the tiled rewrite this file implements -- RATIFIED, not
-- redesigned here).
--
-- 'g_pe_cols' columns process 'g_pe_cols' taps of one input-channel-tile
-- beat per cycle ("group"); 'g_pe_rows' rows compute 'g_pe_rows' output
-- channels from the *same* 'g_pe_cols' taps, each row using its own weight
-- (broadcast activation, per-lane weight). One beat (one
-- 'g_tile_channels'-channel tile of one output pixel's window) needs
-- 'groups_per_tile = ceil(kernel_h*kernel_w*g_tile_channels / g_pe_cols)'
-- groups, sequenced one per cycle over 'weight_rd_addr'.
--
-- Partial-sum carry (the core of this rewrite): an output pixel needing
-- 'T = ceil(in_channels / g_tile_channels)' input-channel tiles arrives as
-- 'T' consecutive 's_window_m2s' beats, 'first_tile'/'last_tile' marking
-- the first/last of that run (both '1' when 'T = 1', per
-- 'cnn_accel_pkg.window_m2s_t's own contract). 'first_tile = '1'' clears
-- all 'g_pe_rows' accumulators; every beat's group sequence accumulates
-- into them (never cleared between tiles of one pixel); 'last_tile = '1''
-- commits the final int32 sums, once its own group sequence completes, to
-- a one-entry output register and emits 'm_accum_m2s' ('last' mirrors the
-- accepted window's 'last'). The accumulators are 'g_pe_rows' flip-flops
-- holding only the one in-flight pixel's partial sum -- never a whole
-- feature map (doc/cnn_accel_tiled_dataflow_proposal.md section 3's
-- 1.6 MB-avoidance argument).
--
-- 'weight_rd_addr' resets to 0 only on 'first_tile' of a new pixel (see
-- 'weight_base_q' below), then advances continuously, one row per group,
-- across every tile beat of that pixel before wrapping again at the next
-- pixel's 'first_tile' (doc/cnn_accel_tiled_dataflow_proposal.md section 4).
--
-- Only one window (one tile beat) is ever "in flight" through the compute
-- engine; the completed-but-undrained pixel result is held in a separate
-- one-entry output register (proposal doc section 3.6) so the compute
-- engine can already start the next tile beat's (or next pixel's) group
-- sequence while the previous pixel's result waits on 'm_accum_s2m.ready'.
--
-- MAC pipeline (S7 timing fix, flow_status.md): the multiply-then-
-- accumulate for one group is split over a register pipeline instead of
-- one combinational cone, because the single-cycle version's critical path
-- (window register -> tap mux -> int8xint8 multiply -> 'g_pe_cols' chained
-- 'g_accum_width'-bit adds -> accumulator register, 31 logic levels /
-- 17.6 ns measured by Vivado synthesis) capped this entity at ~56 MHz,
-- roughly a third of the 150 MHz target:
--
--   stage 0 ('run', address issue): drive 'weight_rd_addr' for group 'g'
--     and latch that group's 'g_pe_cols' activation taps (masked to zero
--     beyond 'mac_taps') out of 'window_q' into 'tap_q';
--   stage 1: 'weight_rd_data' for group 'g' has arrived (the weight
--     buffer's own 1-cycle read latency), so multiply it lane-wise with
--     'tap_q' into 'prod_q' -- one 16-bit product per PE cell;
--   stages 2 .. 1+c_tree_levels: reduce each PE row's 'g_pe_cols'
--     products with a *balanced* 'c_psum_width'-bit adder tree, one tree
--     level per register stage ('reduce_q');
--   final stage: one 'g_accum_width'-bit add of the reduced per-group sum
--     into 'accum_q'.
--
-- Total 'c_mac_latency = 2 + c_tree_levels' cycles from a group's address
-- issue to its contribution landing in 'accum_q' (5 at the project's
-- 'g_pe_cols = 8'). Throughput is unchanged at one group per cycle.
--
-- Because the pipeline is self-timed off its own valid flags rather than
-- off 'state_q', a *non*-last tile beat returns to 'idle' as soon as its
-- last address is issued and its tail drains into 'accum_q' while the next
-- tile beat of the same pixel is already issuing (both only ever add into
-- the same running sum, one entry per cycle, so the overlap is safe). Only
-- a 'last_tile' beat has to wait, in the new 'drain' state, for its own
-- last group to reach the accumulate stage before the pixel's sum can be
-- committed -- 'c_mac_latency' cycles once per output pixel, against one
-- cycle per *beat* saved by the earlier 'idle' return. Net per-pixel cost
-- is 'T*(num_groups + 1) + c_mac_latency' cycles versus the previous
-- 'T*(num_groups + 2)': cheaper for every pixel with more than
-- 'c_mac_latency' input-channel tiles (which is most of the target
-- backbone -- T runs 1, 2, 4, 4, 8, 8, 16, 16, 32), and
-- 'doc/cnn_accel_sizing_proposal.md' section 3's frame model (which counts
-- only the 'T*num_groups' group-issue cycles) is unaffected either way.
--
-- v1 scope deviation from the pre-tiling requirement/proposal docs
-- (recorded prominently per this round's instructions, not silently
-- dropped): 'cfg_opcode'/'cfg_in_channels'/'cfg_out_channels' are NOT
-- ports of this entity. doc/cnn_accel_tiled_dataflow_proposal.md section 3
-- (the ratified design for this rewrite) defines only the broadcast-
-- activation/per-lane-weight 'CONV2D'/'FC' compute grouping -- it does not
-- mention 'DWCONV2D' or a per-opcode indexing mode at all, and how
-- 'DWCONV2D's "no cross-channel accumulation" would even interact with
-- channel tiling (which channel does each tile's beat correspond to?) is
-- not specified anywhere the tiling rewrite was ratified. Implementing it
-- here would be redesigning, not implementing, the ratified doc.
-- 'cfg_in_channels' is also no longer needed for MAC sequencing: channel
-- tiling (T beats/pixel, D11 zero-padding of a partial final tile) is
-- entirely upstream, in 'cnn_accel_window_gen'; 'groups_per_tile' depends
-- only on runtime 'cfg_kernel_h'/'cfg_kernel_w' and the fixed
-- 'g_tile_channels' generic. DWCONV2D support (and the resulting
-- 'cfg_opcode' port) is left as an explicit open item for a future round,
-- once its interaction with channel tiling has an owner and a ratified
-- design -- see the final report for this milestone.
entity cnn_accel_pe_array is
  generic (
    -- Output-channel parallelism: number of parallel accumulator lanes.
    g_pe_rows : positive;
    -- Input-channel/MAC parallelism per cycle: weight-buffer columns
    -- (and window taps) consumed per group.
    g_pe_cols : positive;
    -- Accumulator width (int32 default).
    g_accum_width : positive := 32;
    -- Upper bound on 'cfg_kernel_h'/'cfg_kernel_w' individually; also
    -- sizes 's_window_m2s.data' together with 'g_tile_channels' (must
    -- match the 'cnn_accel_window_gen' instance feeding this port).
    g_max_kernel_size : positive;
    -- Input channels per window-generator tile ("Ct"). Must match the
    -- 'cnn_accel_window_gen' instance feeding this port
    -- (doc/cnn_accel_tiled_dataflow_proposal.md section 1/3).
    g_tile_channels : positive;
    -- Rows in the active cnn_accel_weight_buffer bank. Added by vhdesign
    -- (not in the requirement) so 'weight_rd_addr' can be sized to match
    -- cnn_accel_weight_buffer's actual 'weight_rd_addr' width
    -- (num_bits_needed(g_weight_buffer_depth - 1)) at cnn_accel_top
    -- integration time -- see proposal doc section 3.1, same precedent
    -- cnn_accel_bias_requant's 'g_bias_addr_width' already set. Must cover
    -- the whole layer's rows ('T * groups_per_tile'), per
    -- doc/cnn_accel_tiled_dataflow_proposal.md section 4.
    g_weight_buffer_depth : positive
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top
    -- level): clears the FSM to 'idle' and drops any in-flight/pending
    -- accumulation and any pending-but-undrained output beat. The
    -- accumulator registers' content does not need explicit reset (only
    -- ever read while the, already-reset, state machine guarantees a
    -- fresh 'first_tile' beat clears them before any post-reset
    -- accumulate) -- see the entity-level comment.
    reset : in std_ulogic := '0';
    --# {{}}
    -- Kernel height/width for the in-flight layer. Added by vhdesign (not
    -- in the requirement's port list, per pe_array_proposal.md section
    -- 3.2) -- same port widths/types as cnn_accel_window_gen's own
    -- 'cfg_kernel_h'/'cfg_kernel_w'. Sampled combinationally at
    -- 's_window' accept time; latched for the whole (multi-cycle) group
    -- sequence of that beat.
    cfg_kernel_h : in std_ulogic_vector(7 downto 0);
    cfg_kernel_w : in std_ulogic_vector(7 downto 0);
    --# {{}}
    -- One input-channel-tile window per beat, from cnn_accel_window_gen.
    -- 'data' element 'i' (row-major spatial tap 't = i / g_tile_channels',
    -- channel 'c = i mod g_tile_channels') is tap 't', channel 'c' of this
    -- tile -- unmodified pass-through of cnn_accel_window_gen's own
    -- element layout (cnn_accel_pkg.vhd's 'window_m2s_t' doc comment).
    -- 'first_tile'/'last_tile' mark the first/last of the 'T' tile beats
    -- of the current output pixel (both '1' when 'T = 1').
    s_window_m2s : in window_m2s_t(data(0 to window_data_length(g_max_kernel_size, g_tile_channels) - 1));
    s_window_s2m : out window_s2m_t;
    --# {{}}
    -- Row (tile) address into the active cnn_accel_weight_buffer bank's
    -- weight region. Sequenced 0 .. num_groups-1 per beat, continuing
    -- (not restarting) across every tile beat of one output pixel, and
    -- restarting at 0 only on the next pixel's 'first_tile' beat -- see
    -- the entity-level comment and proposal doc section 4. Driven
    -- unconditionally by this module's own sequencing counters (not
    -- AXI4-Stream: a simple, always-ready, 1-cycle-latency read port).
    weight_rd_addr : out std_ulogic_vector(num_bits_needed(g_weight_buffer_depth - 1) - 1 downto 0);
    -- One int8 weight per PE lane ('g_pe_rows*g_pe_cols' lanes), lane
    -- 'l = r*g_pe_cols + c' (row-major) at bits '8*(l+1)-1 downto 8*l'.
    -- Registered, 1 cycle read latency (cnn_accel_weight_buffer's own
    -- contract).
    weight_rd_data : in std_ulogic_vector(8 * g_pe_rows * g_pe_cols - 1 downto 0);
    --# {{}}
    -- One int32 (g_accum_width-bit) accumulator per output-channel lane
    -- ('g_pe_rows' lanes), one beat per output pixel, emitted once that
    -- pixel's 'last_tile' beat's group sequence completes. 'last' mirrors
    -- the accepted window's 'last'. To cnn_accel_bias_requant.
    m_accum_m2s : out accum_m2s_t(data(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0));
    m_accum_s2m : in accum_s2m_t
  );
end entity cnn_accel_pe_array;

architecture a of cnn_accel_pe_array is

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  -- Element count of 's_window_m2s.data' (elaboration-time upper bound,
  -- from 'g_max_kernel_size'; the runtime-actual tap count per beat,
  -- 'cfg_kernel_h*cfg_kernel_w*g_tile_channels', is <= this).
  constant c_window_len : positive := window_data_length(g_max_kernel_size, g_tile_channels);

  -- Upper bound on 'groups_per_tile' (runtime 'mac_taps' is <= c_window_len).
  constant c_max_groups_per_tile : positive := (c_window_len + g_pe_cols - 1) / g_pe_cols;

  constant c_addr_width : positive := num_bits_needed(g_weight_buffer_depth - 1);
  -- One extra bit over 'c_addr_width', so the "does this pixel's weight
  -- address sequence still fit" bound check (below) never wraps around.
  constant c_ptr_width : positive := num_bits_needed(g_weight_buffer_depth);

  -- int8 x int8 product width, and the balanced-tree reduction that sums
  -- one PE row's 'g_pe_cols' products into a single per-group partial sum.
  -- 'c_tree_width' rounds 'g_pe_cols' up to a power of two so the
  -- reduction loops are fully static (no variable loop bound, no dynamic
  -- slice -- this file's own house rules); the padding entries are
  -- constant zero and optimize away. 'c_psum_width' is the exact width a
  -- sum of 'g_pe_cols' int8 x int8 products needs.
  constant c_product_width : positive := 16;
  constant c_tree_levels : positive := num_bits_needed(g_pe_cols - 1);
  constant c_tree_width : positive := 2 ** c_tree_levels;
  constant c_psum_width : positive := c_product_width + c_tree_levels;

  -- Index of the last reduction stage (the one whose entry 0 is a
  -- complete per-group partial sum).
  constant c_last_reduce : natural := c_tree_levels - 1;

  -- Cycles from a group's 'weight_rd_addr' issue to its contribution
  -- landing in 'accum_q': 1 weight-buffer read latency + 1 multiply
  -- stage + 'c_tree_levels' reduction stages + 1 accumulate stage, minus
  -- the issue cycle itself. Documentation only -- the 'drain' state below
  -- is self-timed off 'reduce_final_q', not off this number.
  constant c_mac_latency : positive := 2 + c_tree_levels;

  -- 'drain' (new with the MAC pipeline, entity-level comment): a
  -- 'last_tile' beat has issued its last group's address but that group is
  -- still in flight through the multiply/reduce/accumulate stages, so the
  -- pixel's sum is not complete yet. Blocks 's_window_s2m.ready' for
  -- exactly 'c_mac_latency' cycles, which is also what keeps the next
  -- pixel's 'first_tile' accumulator clear from ever racing an in-flight
  -- partial sum.
  type state_t is (idle, run, drain, done);
  signal state_q : state_t := idle;

  -- Per-beat latched configuration/data, captured at 's_window' accept
  -- time and held stable for that beat's whole multi-cycle group
  -- sequence (pe_array_proposal.md section 3.2).
  signal window_q : tap_array_t(0 to c_window_len - 1) := (others => (others => '0'));
  signal window_last_q : std_ulogic := '0';
  signal last_tile_q : std_ulogic := '0';
  signal mac_taps_q : natural range 0 to c_window_len := 0;
  signal num_groups_q : natural range 0 to c_max_groups_per_tile := 0;

  -- Group-sequencing counter: runs 0 .. num_groups_q - 1, one weight-row
  -- address issued per cycle ('num_groups_q' cycles in 'run' per beat).
  -- The MAC result of group 'cycle_q' lands in 'accum_q' three cycles
  -- later, via the stage pipeline below.
  signal cycle_q : natural range 0 to c_max_groups_per_tile := 0;

  -- Weight-row base for the *current pixel*: 0 on the pixel's 'first_tile'
  -- beat, advanced by that beat's own 'num_groups_q' every time a beat's
  -- group sequence completes -- so it always points at the next tile
  -- beat's first row (doc/cnn_accel_tiled_dataflow_proposal.md section 4).
  signal weight_base_q : unsigned(c_ptr_width - 1 downto 0) := (others => '0');

  -- Partial-sum accumulators: 'g_pe_rows' flip-flops, cleared on
  -- 'first_tile', carried across every tile beat of one pixel, holding
  -- only the one in-flight pixel's sums (entity-level comment).
  signal accum_q : accum_array_t(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0) :=
    (others => (others => '0'));

  -- One-entry output register, decoupled from the compute engine
  -- (proposal doc section 3.6): the compute engine can already start the
  -- next tile beat's (or next pixel's) group sequence while a previous
  -- pixel's result still waits here on 'm_accum_s2m.ready'.
  signal out_valid_q : std_ulogic := '0';
  signal out_data_q : accum_array_t(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0) :=
    (others => (others => '0'));
  signal out_last_q : std_ulogic := '0';

  ------------------------------------------------------------------------
  -- MAC pipeline registers (entity-level comment). One group is in each
  -- stage at a time; 'valid'/'final' flags travel with it so stages 1-3
  -- run off their own pipeline rather than off 'state_q' (that is what
  -- lets a non-last tile beat's tail drain while the next beat is already
  -- issuing).
  ------------------------------------------------------------------------

  -- Stage 0 output: this group's 'g_pe_cols' activation taps, broadcast to
  -- every PE row, already masked to zero for lanes at/beyond 'mac_taps_q'
  -- (the masking the old combinational MAC did between multiply and add --
  -- moved ahead of the multiply, where it is a select on the tap mux that
  -- has to happen anyway, instead of a gate in the accumulate path).
  signal tap_q : tap_array_t(0 to g_pe_cols - 1) := (others => (others => '0'));
  signal tap_valid_q : std_ulogic := '0';
  -- 'this is the last group of a last_tile beat', i.e. the group whose
  -- stage-3 accumulate completes the output pixel.
  signal tap_final_q : std_ulogic := '0';

  -- Stage 1 output: one int8 x int8 product per PE cell (row 'r' reads
  -- weight lane 'r*g_pe_cols + c' -- per-lane weight, unchanged indexing).
  type prod_row_t is array (0 to g_pe_cols - 1) of signed(c_product_width - 1 downto 0);
  type prod_array_t is array (0 to g_pe_rows - 1) of prod_row_t;
  signal prod_q : prod_array_t := (others => (others => (others => '0')));
  signal prod_valid_q : std_ulogic := '0';
  signal prod_final_q : std_ulogic := '0';

  -- Stages 2 .. 2+c_last_reduce: the balanced adder-tree reduction of one
  -- PE row's products, *one tree level per register stage*. The original
  -- combinational MAC summed the products with a chain
  -- ('result(r) := result(r) + product', 'g_pe_cols' g_accum_width-bit
  -- adds back to back) and reached 56 MHz; collapsing that chain into a
  -- single-stage balanced tree got 110 MHz, but the whole tree in one
  -- stage was then itself every one of the worst endpoints (measured:
  -- 18 logic levels, 11 series CARRY4, 9.1 ns 'prod_q' -> tree output),
  -- still short of the 150 MHz target -- hence one register per level.
  --
  -- 'reduce_q(level)(r)(i)' holds the result of tree level 'level' for PE
  -- row 'r'. Only the '2**(level+1)'-aligned entries are live at a given
  -- level, so 'reduce_q(c_last_reduce)(r)(0)' is row 'r's complete
  -- per-group partial sum; the dead entries are never read and synthesis
  -- trims them. Declaring the full cube (rather than a per-level width)
  -- is what keeps the reduction loops static and generic in 'g_pe_cols'.
  type tree_level_t is array (0 to c_tree_width - 1) of signed(c_psum_width - 1 downto 0);
  type tree_rows_t is array (0 to g_pe_rows - 1) of tree_level_t;
  type tree_pipe_t is array (0 to c_last_reduce) of tree_rows_t;
  signal reduce_q : tree_pipe_t := (others => (others => (others => (others => '0'))));
  signal reduce_valid_q : std_ulogic_vector(0 to c_last_reduce) := (others => '0');
  signal reduce_final_q : std_ulogic_vector(0 to c_last_reduce) := (others => '0');

  -- Zero-padded, width-extended read of PE row product 'i', for the
  -- tree's power-of-two padding entries (only reachable when 'g_pe_cols'
  -- is not itself a power of two). 'i' is a loop constant at every
  -- unrolled call site, so the branch resolves at elaboration time.
  function padded_product(row : prod_row_t; i : natural) return signed is
  begin
    if i < g_pe_cols then
      return resize(row(i), c_psum_width);
    end if;
    return to_signed(0, c_psum_width);
  end function;

begin

  ------------------------------------------------------------------------
  -- Handshake: 's_window_s2m.ready' is a pure function of registered
  -- state ('state_q = idle'), never of 's_window_m2s.valid' itself -- no
  -- combinational loop (shared/Axi4.md). 'weight_rd_addr' is likewise a
  -- pure function of registered state, driven unconditionally while a
  -- group remains to be issued this beat (proposal doc section 10).
  ------------------------------------------------------------------------

  s_window_s2m.ready <= '1' when state_q = idle else '0';

  weight_rd_addr <=
    std_ulogic_vector(
      resize(weight_base_q + to_unsigned(cycle_q, c_ptr_width), c_addr_width)
    ) when state_q = run else
    (others => '0');

  m_accum_m2s.valid <= out_valid_q;
  m_accum_m2s.data <= out_data_q;
  m_accum_m2s.last <= out_last_q;

  ------------------------------------------------------------------------
  -- Compute engine: window-tile accept (idle) -> address issue (run) ->
  -- pipeline drain (drain, 'last_tile' beats only) -> output-register
  -- wait (done, only entered when a 'last_tile' beat's commit found the
  -- output register still full) -- proposal doc section 5, extended with
  -- the multi-tile partial-sum carry (doc/cnn_accel_tiled_dataflow_
  -- proposal.md section 3) and the three-stage MAC pipeline (this file's
  -- entity-level comment).
  ------------------------------------------------------------------------

  main : process(clk)
    variable kernel_h_v : natural;
    variable kernel_w_v : natural;
    variable mac_taps_v : natural;
    variable num_groups_v : natural;
    variable effective_base_v : natural;
    variable idx_v : natural;
    variable weight_lane_v : natural;
    variable final_accum_v : accum_array_t(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0);
    variable can_commit_v : boolean;
  begin
    if rising_edge(clk) then
      -- Stage 3 (the only place 'accum_q' is ever added into): one
      -- 'g_accum_width'-bit add per PE row. Computed here as a variable so
      -- the 'drain' commit below can use the same value in the same cycle
      -- it is written back (no extra commit cycle).
      for r in 0 to g_pe_rows - 1 loop
        final_accum_v(r) :=
          accum_q(r) + resize(reduce_q(c_last_reduce)(r)(0), g_accum_width);
      end loop;

      if reset = '1' then
        state_q <= idle;
        cycle_q <= 0;
        weight_base_q <= (others => '0');
        out_valid_q <= '0';
        tap_valid_q <= '0';
        prod_valid_q <= '0';
        reduce_valid_q <= (others => '0');
      else
        can_commit_v := (out_valid_q = '0') or (m_accum_s2m.ready = '1');

        -- Default output-register drain, overridden below by a same-
        -- cycle commit (no bubble on back-to-back commit+drain) -- same
        -- idiom as cnn_accel_pool.vhd/cnn_accel_bias_requant.vhd's own
        -- one-entry output registers.
        if out_valid_q = '1' and m_accum_s2m.ready = '1' then
          out_valid_q <= '0';
        end if;

        --------------------------------------------------------------
        -- MAC pipeline stages 3, 2 and 1. Driven purely by their own
        -- valid pipeline, never by 'state_q', so a non-'last_tile'
        -- beat's tail keeps draining into 'accum_q' after the FSM has
        -- already returned to 'idle' and accepted the next tile beat of
        -- the same pixel (entity-level comment). Written before the
        -- 'case' so the 'first_tile' accumulator clear below overrides
        -- the stage-3 write-back -- which is safe precisely because a
        -- 'first_tile' beat can only be accepted with the pipeline
        -- empty (the previous pixel's 'drain' state guarantees it).
        --------------------------------------------------------------

        if reduce_valid_q(c_last_reduce) = '1' then
          accum_q <= final_accum_v;
        end if;

        -- Reduction stage 0: pair up this row's products. Only the
        -- even-indexed entries are written (and read at the next level);
        -- 'padded_product' supplies a constant zero for the power-of-two
        -- padding lanes.
        for r in 0 to g_pe_rows - 1 loop
          for i in 0 to c_tree_width - 1 loop
            if i mod 2 = 0 then
              reduce_q(0)(r)(i) <=
                padded_product(prod_q(r), i) + padded_product(prod_q(r), i + 1);
            end if;
          end loop;
        end loop;
        reduce_valid_q(0) <= prod_valid_q;
        reduce_final_q(0) <= prod_final_q;

        -- Reduction stages 1 .. c_last_reduce: one balanced tree level
        -- each. Null range when 'g_pe_cols' <= 2.
        for level in 1 to c_last_reduce loop
          for r in 0 to g_pe_rows - 1 loop
            for i in 0 to c_tree_width - 1 loop
              if i mod (2 ** (level + 1)) = 0 then
                reduce_q(level)(r)(i) <=
                  reduce_q(level - 1)(r)(i) + reduce_q(level - 1)(r)(i + 2 ** level);
              end if;
            end loop;
          end loop;
          reduce_valid_q(level) <= reduce_valid_q(level - 1);
          reduce_final_q(level) <= reduce_final_q(level - 1);
        end loop;

        for r in 0 to g_pe_rows - 1 loop
          for c in 0 to g_pe_cols - 1 loop
            weight_lane_v := r * g_pe_cols + c;
            prod_q(r)(c) <=
              signed(tap_q(c))
              * signed(weight_rd_data(8 * (weight_lane_v + 1) - 1 downto 8 * weight_lane_v));
          end loop;
        end loop;
        prod_valid_q <= tap_valid_q;
        prod_final_q <= tap_final_q;

        -- Stage 0 default: no group issued this cycle (overridden in
        -- 'run' below).
        tap_valid_q <= '0';
        tap_final_q <= '0';

        case state_q is

          when idle =>
            if s_window_m2s.valid = '1' then
              kernel_h_v := to_integer(unsigned(cfg_kernel_h));
              kernel_w_v := to_integer(unsigned(cfg_kernel_w));

              assert kernel_h_v <= g_max_kernel_size and kernel_w_v <= g_max_kernel_size
                report "cnn_accel_pe_array: cfg_kernel_h/cfg_kernel_w must be <= g_max_kernel_size"
                severity failure;

              mac_taps_v := kernel_h_v * kernel_w_v * g_tile_channels;

              assert mac_taps_v >= 1
                report "cnn_accel_pe_array: cfg_kernel_h/cfg_kernel_w must both be >= 1"
                severity failure;

              assert mac_taps_v <= c_window_len
                report "cnn_accel_pe_array: kernel_h*kernel_w*g_tile_channels (" &
                  natural'image(mac_taps_v) & ") exceeds s_window_m2s.data's element count (" &
                  natural'image(c_window_len) & ")"
                severity failure;

              num_groups_v := (mac_taps_v + g_pe_cols - 1) / g_pe_cols;

              if s_window_m2s.first_tile = '1' then
                effective_base_v := 0;
              else
                effective_base_v := to_integer(weight_base_q);
              end if;

              assert effective_base_v + num_groups_v <= g_weight_buffer_depth
                report "cnn_accel_pe_array: weight_rd_addr sequence (base " &
                  natural'image(effective_base_v) & " + " & natural'image(num_groups_v) &
                  " groups) exceeds g_weight_buffer_depth (" &
                  natural'image(g_weight_buffer_depth) & ")"
                severity failure;

              if s_window_m2s.first_tile = '1' then
                weight_base_q <= (others => '0');
                -- Partial-sum carry: a first-tile beat starts a fresh
                -- pixel, so the accumulators from any previous pixel are
                -- cleared here rather than after this beat's own group
                -- sequence -- the same cycle-1 group's partial sum is
                -- added into an all-zero accumulator (see the 'run'
                -- state below, which always adds into 'accum_q').
                accum_q <= (others => (others => '0'));
              end if;

              window_q <= s_window_m2s.data;
              window_last_q <= s_window_m2s.last;
              last_tile_q <= s_window_m2s.last_tile;
              mac_taps_q <= mac_taps_v;
              num_groups_q <= num_groups_v;
              cycle_q <= 0;
              state_q <= run;
            end if;

          when run =>
            -- Stage 0: issue group 'cycle_q's weight-row address (the
            -- concurrent 'weight_rd_addr' assignment above) and latch that
            -- group's activation taps. Tap 'idx = cycle_q*g_pe_cols + c'
            -- is broadcast to every PE row; lanes at/beyond 'mac_taps_q'
            -- are masked to zero here (the last group of a beat whose
            -- 'mac_taps' is not a multiple of 'g_pe_cols'), which is also
            -- what keeps the index inside 'window_q's range -- an
            -- out-of-range but masked-invalid index must never raise a
            -- VHDL range error. Both loop bounds are generics, so this
            -- unrolls into a static mux per column: no dynamic slice, no
            -- variable-bound loop (this file's house rules,
            -- cnn_accel_pkg.vhd's 'window_m2s_t' doc comment).
            for c in 0 to g_pe_cols - 1 loop
              idx_v := cycle_q * g_pe_cols + c;
              if idx_v < mac_taps_q then
                tap_q(c) <= window_q(idx_v);
              else
                tap_q(c) <= (others => '0');
              end if;
            end loop;
            tap_valid_q <= '1';

            if cycle_q = num_groups_q - 1 then
              -- Last address of this beat. 'weight_base_q' advances by
              -- this beat's own 'num_groups_q' so the next tile beat of
              -- this pixel continues (not restarts) the weight-row
              -- sequence -- doc/cnn_accel_tiled_dataflow_proposal.md
              -- section 4, unchanged by the pipeline.
              tap_final_q <= last_tile_q;
              weight_base_q <= weight_base_q + to_unsigned(num_groups_q, c_ptr_width);
              cycle_q <= 0;

              if last_tile_q = '1' then
                -- The pixel's sum is not complete until this group
                -- reaches stage 3, three cycles from now.
                state_q <= drain;
              else
                -- Nothing left to wait for: the next tile beat of this
                -- pixel can be accepted immediately and its own groups
                -- will simply queue up behind this beat's tail, adding
                -- into the same running 'accum_q'.
                state_q <= idle;
              end if;
            else
              cycle_q <= cycle_q + 1;
            end if;

          when drain =>
            -- 'reduce_final_q(c_last_reduce)' marks the 'last_tile' beat's
            -- last group arriving at the accumulate stage, so
            -- 'final_accum_v' *is* this output pixel's completed sum this
            -- cycle (and is written back into 'accum_q' by the accumulate
            -- block above, which is what the 'done' path below then
            -- commits).
            if reduce_valid_q(c_last_reduce) = '1'
              and reduce_final_q(c_last_reduce) = '1' then
              if can_commit_v then
                out_valid_q <= '1';
                out_data_q <= final_accum_v;
                out_last_q <= window_last_q;
                state_q <= idle;
              else
                state_q <= done;
              end if;
            end if;

          when done =>
            if can_commit_v then
              out_valid_q <= '1';
              out_data_q <= accum_q;
              out_last_q <= window_last_q;
              state_q <= idle;
            end if;

        end case;
      end if;
    end if;
  end process;

end architecture a;
