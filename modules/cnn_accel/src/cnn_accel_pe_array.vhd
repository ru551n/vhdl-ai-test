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

  type state_t is (idle, run, done);
  signal state_q : state_t := idle;

  -- Per-beat latched configuration/data, captured at 's_window' accept
  -- time and held stable for that beat's whole multi-cycle group
  -- sequence (pe_array_proposal.md section 3.2).
  signal window_q : tap_array_t(0 to c_window_len - 1) := (others => (others => '0'));
  signal window_last_q : std_ulogic := '0';
  signal last_tile_q : std_ulogic := '0';
  signal mac_taps_q : natural range 0 to c_window_len := 0;
  signal num_groups_q : natural range 0 to c_max_groups_per_tile := 0;

  -- Group-sequencing counter: runs 0 .. num_groups_q inclusive
  -- ('num_groups_q + 1' cycles total per beat -- pe_array_proposal.md
  -- section 5).
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
  -- Per-lane MAC for one group (proposal doc section 6, shape unchanged
  -- by tiling): PE row 'r', column 'c', group 'g' (the group whose
  -- weights are valid on 'weight_data' this cycle) reads window tap
  -- 'idx = g*g_pe_cols + c' (broadcast activation -- every row reads the
  -- same tap) and weight lane 'r*g_pe_cols + c' (per-lane weight),
  -- multiplies, and returns the (masked) per-lane partial sum to be added
  -- into that group's accumulators. 'idx' is clamped into 'window''s
  -- actual range before the read (a defensive measure: the last group of
  -- a beat whose 'mac_taps' does not divide evenly by 'g_pe_cols' can
  -- compute an 'idx' at/beyond 'mac_taps', and 'window''s own bound
  -- ('c_window_len') can be smaller than 'g_pe_cols' rounds up to, when
  -- 'g_pe_cols' does not divide 'c_window_len' evenly either) -- an
  -- out-of-range but masked-invalid 'idx' must never raise a VHDL range
  -- error, and the 'r'/'c' loops are constant-bound (generics), so this
  -- whole function unrolls at elaboration/synthesis time: no dynamic
  -- slice, no variable-bound loop (see this file's own house rules and
  -- cnn_accel_pkg.vhd's 'window_m2s_t' doc comment on array indexing).
  ------------------------------------------------------------------------

  function compute_partial_sums(
    window : tap_array_t;
    weight_data : std_ulogic_vector;
    mac_group : natural;
    mac_taps : natural
  ) return accum_array_t is
    variable result : accum_array_t(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0) :=
      (others => (others => '0'));
    variable idx : natural;
    variable idx_clamped : natural;
    variable is_valid : boolean;
    variable operand : signed(7 downto 0);
    variable weight_lane : natural;
    variable weight : signed(7 downto 0);
    variable product : signed(15 downto 0);
  begin
    for r in 0 to g_pe_rows - 1 loop
      for c in 0 to g_pe_cols - 1 loop
        idx := mac_group * g_pe_cols + c;
        is_valid := idx < mac_taps;

        if idx <= window'high then
          idx_clamped := idx;
        else
          idx_clamped := window'high;
        end if;

        operand := signed(window(idx_clamped));

        weight_lane := r * g_pe_cols + c;
        weight := signed(weight_data(8 * (weight_lane + 1) - 1 downto 8 * weight_lane));

        product := operand * weight;

        if is_valid then
          result(r) := result(r) + resize(product, g_accum_width);
        end if;
      end loop;
    end loop;
    return result;
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
    ) when (state_q = run and cycle_q < num_groups_q) else
    (others => '0');

  m_accum_m2s.valid <= out_valid_q;
  m_accum_m2s.data <= out_data_q;
  m_accum_m2s.last <= out_last_q;

  ------------------------------------------------------------------------
  -- Compute engine: window-tile accept (idle) -> group sequencing (run)
  -- -> output-register wait (done, only entered when a 'last_tile' beat's
  -- commit found the output register still full) -- proposal doc
  -- section 5, extended with the multi-tile partial-sum carry (this
  -- file's core rewrite, doc/cnn_accel_tiled_dataflow_proposal.md
  -- section 3).
  ------------------------------------------------------------------------

  main : process(clk)
    variable kernel_h_v : natural;
    variable kernel_w_v : natural;
    variable mac_taps_v : natural;
    variable num_groups_v : natural;
    variable effective_base_v : natural;
    variable partial_v : accum_array_t(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0);
    variable final_accum_v : accum_array_t(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0);
    variable can_commit_v : boolean;
  begin
    if rising_edge(clk) then
      if reset = '1' then
        state_q <= idle;
        cycle_q <= 0;
        weight_base_q <= (others => '0');
        out_valid_q <= '0';
      else
        can_commit_v := (out_valid_q = '0') or (m_accum_s2m.ready = '1');

        -- Default output-register drain, overridden below by a same-
        -- cycle commit (no bubble on back-to-back commit+drain) -- same
        -- idiom as cnn_accel_pool.vhd/cnn_accel_bias_requant.vhd's own
        -- one-entry output registers.
        if out_valid_q = '1' and m_accum_s2m.ready = '1' then
          out_valid_q <= '0';
        end if;

        case state_q is

          when idle =>
            if s_window_m2s.valid = '1' then
              kernel_h_v := to_integer(unsigned(cfg_kernel_h));
              kernel_w_v := to_integer(unsigned(cfg_kernel_w));

              assert kernel_h_v <= g_max_kernel_size and kernel_w_v <= g_max_kernel_size
                report "cnn_accel_pe_array: cfg_kernel_h/cfg_kernel_w must be <= g_max_kernel_size"
                severity failure;

              mac_taps_v := kernel_h_v * kernel_w_v * g_tile_channels;

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
            if cycle_q >= 1 then
              partial_v := compute_partial_sums(
                window => window_q,
                weight_data => weight_rd_data,
                mac_group => cycle_q - 1,
                mac_taps => mac_taps_q
              );
            else
              partial_v := (others => (others => '0'));
            end if;

            for r in 0 to g_pe_rows - 1 loop
              final_accum_v(r) := accum_q(r) + partial_v(r);
            end loop;

            if cycle_q = num_groups_q then
              -- This beat's group sequence is done: 'final_accum_v' is
              -- this pixel's running sum through (and including) this
              -- tile. Always carried into 'accum_q' -- needed by the next
              -- tile beat of this pixel when 'last_tile_q = '0'',
              -- harmless (overwritten by the next pixel's 'first_tile'
              -- clear) otherwise.
              accum_q <= final_accum_v;
              weight_base_q <= weight_base_q + to_unsigned(num_groups_q, c_ptr_width);

              if last_tile_q = '1' then
                if can_commit_v then
                  out_valid_q <= '1';
                  out_data_q <= final_accum_v;
                  out_last_q <= window_last_q;
                  state_q <= idle;
                else
                  state_q <= done;
                end if;
              else
                state_q <= idle;
              end if;

              cycle_q <= 0;
            else
              accum_q <= final_accum_v;
              cycle_q <= cycle_q + 1;
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
