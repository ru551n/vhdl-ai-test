library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;
use cnn_accel.cnn_accel_isa_pkg.all;

-- Spatial reduction over a 'cnn_accel_window_gen'-produced pooling window.
-- See modules/cnn_accel/doc/cnn_accel_pool_req.md and
-- modules/cnn_accel/doc/cnn_accel_pool_proposal.md.
--
-- 'OPCODE_POOL_MAX': int8 max over the window's active taps, emitted
-- directly on 'm_max' (bypasses 'cnn_accel_bias_requant'). 'OPCODE_POOL_AVG':
-- int32 (g_accum_width-bit) sum over the window's active taps, emitted on
-- 'm_avgsum' for 'cnn_accel_bias_requant' to scale/round (division by the
-- pool area is a downstream requantize op -- see doc/cnn_accel_arch.md
-- "Non-obvious boundary rationale"). The two outputs are structurally
-- mutually exclusive: one shared output register, tagged with which port
-- it belongs to -- see proposal doc section 4.
entity cnn_accel_pool is
  generic (
    -- Upper bound on 'cfg_pool_kernel_h'/'cfg_pool_kernel_w' individually;
    -- sizes the fixed 'g_max_kernel_size**2'-lane tap array/reduction
    -- network. This is the POOL kernel bound
    -- ('cnn_accel_constant_max_pool_kernel_size', 5), deliberately
    -- separate from and larger than the conv datapath's own
    -- 'g_max_kernel_size' (3) -- YOLOv8n's SPPF pools 5x5 while all its
    -- convolutions are 1x1/3x3, so only this path pays for it.
    --
    -- There is no longer an upper bound of the form 'g_max_kernel_size**2
    -- * 8 <= axi_stream_data_sz': 's_window' is the unconstrained
    -- 'window_m2s_t' tap-array record (cnn_accel_pkg) the conv path
    -- already uses, not the fixed-128-bit 'axi_stream_m2s_t' it used to
    -- be. That change is what makes 5x5 (25 taps = 200 bits) possible at
    -- all, and it was made in preference to widening the
    -- hdl-modules-wide 'axi_stream_data_sz', which would have inflated
    -- every stream in the design.
    g_max_kernel_size : positive;
    -- 'OPCODE_POOL_AVG' sum width. Must be wide enough to hold
    -- 'g_max_kernel_size**2 * 127' (the maximum possible window sum)
    -- without overflow -- no saturation/rounding is applied in this
    -- module, per doc/cnn_accel_pool_req.md.
    g_accum_width : positive
  );
  port (
    clk : in std_ulogic;
    -- Synchronous active-high reset ('reset_internal' at the IP top level).
    reset : in std_ulogic := '0';
    --# {{}}
    -- 'OPCODE_POOL_MAX' vs 'OPCODE_POOL_AVG' (cnn_accel_pkg), sampled at
    -- 's_window' accept time -- selects whether that window's result is
    -- routed to 'm_max' or 'm_avgsum'. Any other value is treated as the
    -- 'OPCODE_POOL_MAX' path (upstream routing guarantees this module only
    -- ever receives pooling opcodes).
    cfg_opcode : in std_ulogic_vector(7 downto 0);
    -- Pool kernel height/width for the in-flight instruction. Contract:
    -- '1 <= cfg_pool_kernel_h, cfg_pool_kernel_w <= g_max_kernel_size'.
    -- 'cfg_pool_kernel_h * cfg_pool_kernel_w' taps (the lowest-indexed
    -- elements of 's_window_m2s.data') are active; the rest of the window
    -- beat is ignored.
    cfg_pool_kernel_h : in std_ulogic_vector(7 downto 0);
    cfg_pool_kernel_w : in std_ulogic_vector(7 downto 0);
    --# {{}}
    -- One pooling window per beat, from 'cnn_accel_window_gen' (already
    -- opcode-selected upstream). Element 'i' of 'data' is tap 'i'
    -- (row-major, 'i = row * cfg_pool_kernel_w + col') as a signed int8;
    -- only the 'cfg_pool_kernel_h * cfg_pool_kernel_w' lowest-indexed
    -- elements are active, the rest are ignored (they carry the window
    -- generator's pad value).
    --
    -- 'first_tile'/'last_tile' are unused here: pooling is
    -- channel-parallel across lanes, never channel-tiled, so a pooling
    -- window is always exactly one tile.
    s_window_m2s : in window_m2s_t(data(0 to g_max_kernel_size * g_max_kernel_size - 1));
    s_window_s2m : out window_s2m_t;
    --# {{}}
    -- 'OPCODE_POOL_MAX' result: int8 max over the window's active taps, on
    -- 'data(7 downto 0)' ('data' high bits are 0). To the final output
    -- 'handshake_mux' (bypasses 'cnn_accel_bias_requant').
    m_max_m2s : out axi_stream_m2s_t;
    m_max_s2m : in axi_stream_s2m_t;
    --# {{}}
    -- 'OPCODE_POOL_AVG' result: int32 (g_accum_width-bit) sum over the
    -- window's active taps, on 'data(g_accum_width - 1 downto 0)' ('data'
    -- high bits are 0). To 'cnn_accel_bias_requant' for the pool-area
    -- divide.
    m_avgsum_m2s : out axi_stream_m2s_t;
    m_avgsum_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_pool;

architecture a of cnn_accel_pool is

  function to_sl_local(cond : boolean) return std_ulogic is
  begin
    if cond then
      return '1';
    else
      return '0';
    end if;
  end function;

  function to_integer_sl(value : std_ulogic) return natural is
  begin
    if value = '1' then
      return 1;
    else
      return 0;
    end if;
  end function;

  ------------------------------------------------------------------------
  -- Local sizing constants.
  ------------------------------------------------------------------------

  constant c_max_taps : positive := g_max_kernel_size * g_max_kernel_size;

  type tap_arr_t is array (0 to c_max_taps - 1) of signed(7 downto 0);

  ------------------------------------------------------------------------
  -- Reinterprets the window record's int8 tap array as signed, per the
  -- element layout documented on the 's_window_m2s' port above. A plain
  -- element-wise cast now that 'data' is an array rather than a packed
  -- vector -- no bit slicing left to get wrong.
  ------------------------------------------------------------------------

  function extract_taps(data : tap_array_t) return tap_arr_t is
    variable result : tap_arr_t;
  begin
    for i in 0 to c_max_taps - 1 loop
      result(i) := signed(data(i));
    end loop;
    return result;
  end function;

  -- Smallest 'k' with '2**k >= n' -- the depth of a balanced binary
  -- reduction tree over 'n' leaves, and (plus 8) the width the masked sum
  -- of 'n' int8 taps needs to be carried at without overflow.
  function clog2_ceil(n : positive) return natural is
    variable v_k : natural := 0;
  begin
    while 2 ** v_k < n loop
      v_k := v_k + 1;
    end loop;
    return v_k;
  end function;

  constant c_levels : natural := clog2_ceil(c_max_taps);

  -- 'sum(|tap|) <= c_max_taps * 128 <= 2**c_levels * 128 = 2**(c_levels+7)',
  -- so 'c_levels + 8' bits of two's complement hold every reachable window
  -- sum exactly. The tree therefore carries the sum at this width and
  -- sign-extends once, at the root, to 'g_accum_width' -- identical values
  -- to accumulating at 'g_accum_width' throughout, at a fraction of the
  -- carry-chain length.
  constant c_sum_width : positive := c_levels + 8;

  -- Second tree split point: one level below the root, but never below
  -- the first split. Written as a function rather than
  -- 'max(c_split1, c_levels - 1)' because 'c_levels - 1' would be -1 (a
  -- 'natural' range error at elaboration) for the degenerate
  -- single-tap geometry, which no instantiation uses but which the
  -- expression must still be legal for.
  function split2_of(levels, split1 : natural) return natural is
  begin
    if levels >= 1 and levels - 1 > split1 then
      return levels - 1;
    end if;
    return split1;
  end function;

  -- Number of live nodes at tree level 'level' (level 0 = the taps).
  function level_count(level : natural) return positive is
    variable v_n : positive := c_max_taps;
  begin
    for i in 1 to level loop
      v_n := (v_n + 1) / 2;
    end loop;
    return v_n;
  end function;

  -- The tree is cut by TWO register stages: levels 1 .. c_split1 are
  -- evaluated in the cycle a window beat is accepted, levels c_split1+1 ..
  -- c_split2 in the next, and c_split2+1 .. c_levels in the one after
  -- that. At 5x5 that is two compare/add levels, then two, then one --
  -- instead of five in one cycle, or the two-then-three of the previous
  -- single split. Purely a re-timing of the same tree: no operand, no
  -- operator and no association order changes, so the result is
  -- bit-identical at every split.
  --
  -- Why the second split. With one split the back half carried three
  -- compare levels ('v_max(2) -> v_max(3) -> v_max(4) -> v_max(5)'), and
  -- post-route that was 9 logic levels and 8.49 ns from 'p1_max_q' to
  -- 'out_max_q' -- -1.964 ns at 150 MHz, and the worst path in the whole
  -- accelerator once the ready chains were broken. Splitting the back
  -- half again is the structural fix (shared/TimingAndResources.md,
  -- "Budget logic depth per stage"); rebalancing the single split instead
  -- ('c_split = 3') was measured to be no good, because the FRONT half
  -- starts at the tap mux off 'assembly_q' and already ran at -0.882 ns
  -- with only two levels -- moving a third into it just swaps which half
  -- fails.
  --
  -- Cost: one cycle of latency, no throughput (the elastic control below
  -- still moves one beat per cycle through every stage), and
  -- 'level_count(c_split2)' nodes of extra register -- two of them at
  -- 5x5, i.e. 2 x 8 bits of max plus 2 x 'c_sum_width' bits of sum.
  constant c_split1 : natural := c_levels / 2;
  constant c_split2 : natural := split2_of(c_levels, c_split1);

  type sum_level_t is array (0 to c_max_taps - 1) of signed(c_sum_width - 1 downto 0);
  type sum_tree_t is array (0 to c_levels) of sum_level_t;
  type max_level_t is array (0 to c_max_taps - 1) of signed(7 downto 0);
  type max_tree_t is array (0 to c_levels) of max_level_t;

  ------------------------------------------------------------------------
  -- Combinational reduction over the current 's_window_m2s' beat.
  ------------------------------------------------------------------------

  signal taps : tap_arr_t;
  signal active_count : natural range 0 to c_max_taps;
  signal max_result : signed(7 downto 0);
  signal avgsum_result : signed(g_accum_width - 1 downto 0);
  signal is_avg_sel : std_ulogic;

  ------------------------------------------------------------------------
  -- Per-command tap mask, registered.
  --
  -- 'cfg_pool_kernel_h * cfg_pool_kernel_w' is per-command geometry, and
  -- computing it in the beat datapath -- a runtime 8x8 multiply feeding
  -- the reduction network's lane guards -- was the deepest combinational
  -- cone in the whole accelerator (82 logic levels from 'cmd_proc's
  -- 'desc_q' to 'out_max_q', -49.5 ns at 150 MHz in the first top-level
  -- build). It is now evaluated once into 'tap_mask_q', a plain
  -- one-hot-prefix mask, and the reduction sees only registers.
  --
  -- 'cfg_match' is what keeps that bit-exact without weakening the port
  -- contract: 'tap_mask_q' lags 'cfg_pool_kernel_h/w' by one cycle, so
  -- the cycle in which either changes cannot be allowed to accept a beat.
  -- Deasserting 'ready' for that one cycle (AXI4-Stream legal: 'ready'
  -- may fall at any time) costs a single bubble per kernel-shape change
  -- and nothing at all inside a command, where the descriptor is constant.
  ------------------------------------------------------------------------

  signal tap_mask_q : std_ulogic_vector(0 to c_max_taps - 1) := (others => '0');
  signal cfg_kernel_h_q : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_kernel_w_q : std_ulogic_vector(7 downto 0) := (others => '0');
  signal cfg_match : std_ulogic;

  ------------------------------------------------------------------------
  -- Pipeline stage 1: the partially reduced tree, plus the beat's own
  -- sidebands. 'p1_is_avg_q' is the tag that decides which output port the
  -- result eventually leaves by; it travels with the beat, so a
  -- 'cfg_opcode' change between beats can never retag one already in
  -- flight.
  ------------------------------------------------------------------------

  signal front_max : max_level_t;
  signal front_sum : sum_level_t;

  signal p1_valid_q : std_ulogic := '0';
  signal p1_is_avg_q : std_ulogic := '0';
  signal p1_last_q : std_ulogic := '0';
  signal p1_max_q : max_level_t := (others => (others => '0'));
  signal p1_sum_q : sum_level_t := (others => (others => '0'));

  -- Pipeline stage 2: the tree after levels 'c_split1+1 .. c_split2', with
  -- the same sidebands travelling alongside. See 'c_split2'.
  signal mid_max : max_level_t;
  signal mid_sum : sum_level_t;

  signal p2_valid_q : std_ulogic := '0';
  signal p2_is_avg_q : std_ulogic := '0';
  signal p2_last_q : std_ulogic := '0';
  signal p2_max_q : max_level_t := (others => (others => '0'));
  signal p2_sum_q : sum_level_t := (others => (others => '0'));

  ------------------------------------------------------------------------
  -- Two-entry tagged output buffer, shared by both output streams.
  --
  -- Still one register per result and one tag bit per result -- entry 'e'
  -- is presented on 'm_max' or on 'm_avgsum' according to its own
  -- 'out_is_avg_q(e)', never on both -- so proposal doc section 4's
  -- structural mutual-exclusion argument is unchanged. What changed is the
  -- DEPTH, from one entry to two, and the reason is 'ready':
  --
  --   's_window_s2m.ready' used to be a combinational function of the
  --   selected output port's 'ready'. At the top level that made one
  --   chain out of the ofmap sink's ready, the descriptor's opcode (which
  --   selects between the MAX and AVG output paths), all eight pool
  --   lanes, their AND, and the window generator's launch control -- 14
  --   levels and 10.8 ns of mostly routing, and the largest single group
  --   of failing endpoints in the design.
  --
  --   With two entries, a beat may always be accepted while fewer than
  --   two are buffered, so 'ready' is a function of this entity's own
  --   registers and nothing else. The buffer still empties at one beat per
  --   cycle, so throughput is unchanged; only latency grows, by two
  --   cycles per window.
  ------------------------------------------------------------------------

  constant c_buf_depth : positive := 2;

  type buf_max_t is array (0 to c_buf_depth - 1) of signed(7 downto 0);
  type buf_sum_t is array (0 to c_buf_depth - 1) of signed(g_accum_width - 1 downto 0);

  signal buf_count_q : natural range 0 to c_buf_depth := 0;
  signal out_is_avg_q : std_ulogic_vector(0 to c_buf_depth - 1) := (others => '0');
  signal out_last_q : std_ulogic_vector(0 to c_buf_depth - 1) := (others => '0');
  signal out_max_q : buf_max_t := (others => (others => '0'));
  signal out_avgsum_q : buf_sum_t := (others => (others => '0'));

  -- 'ready' of whichever output port the buffer's HEAD entry targets. Pops
  -- the head; deliberately does NOT reach 's_window_s2m.ready'.
  signal selected_output_ready : std_ulogic;
  signal out_valid : std_ulogic;
  signal pop : std_ulogic;
  -- 'push1' moves stage 1 into stage 2, 'push' moves stage 2 into the
  -- output buffer. Both are functions of registered state only.
  signal push1 : std_ulogic;
  signal push : std_ulogic;
  signal accepted : std_ulogic;

begin

  ------------------------------------------------------------------------
  -- Static sizing contract checks -- see the generics' doc comments.
  ------------------------------------------------------------------------

  -- No tap-count ceiling any more: 's_window' is the unconstrained
  -- 'window_m2s_t' tap array (see the generic's comment). The output
  -- streams are still fixed-width 'axi_stream_m2s_t', so their payloads
  -- are what needs checking -- 'm_max' is one int8 and can never
  -- overflow, 'm_avgsum' is 'g_accum_width' wide.
  assert g_accum_width >= c_sum_width
    report "cnn_accel_pool: g_accum_width is too narrow to hold the widest " &
      "possible window sum (c_levels + 8 bits)"
    severity failure;

  assert g_accum_width <= axi_stream_data_sz
    report "cnn_accel_pool: g_accum_width exceeds axi_stream_data_sz " &
      "(the fixed-width axi_stream_pkg data field cannot carry that wide a sum)"
    severity failure;

  ------------------------------------------------------------------------
  -- Reduction (combinational).
  ------------------------------------------------------------------------

  taps <= extract_taps(s_window_m2s.data);
  active_count <= to_integer(unsigned(cfg_pool_kernel_h)) * to_integer(unsigned(cfg_pool_kernel_w));
  is_avg_sel <= '1' when cfg_opcode = OPCODE_POOL_AVG else '0';

  cfg_match <= '1'
    when cfg_pool_kernel_h = cfg_kernel_h_q and cfg_pool_kernel_w = cfg_kernel_w_q
    else '0';

  -- Resetless: 'tap_mask_q' is configuration, and it tracks the config
  -- ports unconditionally, so one cycle after reset releases it is already
  -- consistent with them (and 'cfg_match' reports as much).
  config_stage : process(clk)
  begin
    if rising_edge(clk) then
      cfg_kernel_h_q <= cfg_pool_kernel_h;
      cfg_kernel_w_q <= cfg_pool_kernel_w;
      for i in 0 to c_max_taps - 1 loop
        tap_mask_q(i) <= '1' when i < active_count else '0';
      end loop;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Masked balanced reduction trees.
  --
  -- Bit-exact against the linear reductions they replace: an inactive lane
  -- contributes the identity of its operator (-128 for max, which is also
  -- the linear version's seed and the minimum int8, and 0 for the sum), and
  -- both operators are associative and commutative over the values that can
  -- reach them, so re-associating the 'c_max_taps' lanes into a tree cannot
  -- change the result. The sum carries 'c_sum_width' bits, proven above to
  -- be free of overflow, and is sign-extended once at the root.
  --
  -- Depth is 'c_levels' (5 for a 5x5 pool) instead of 'c_max_taps' (25).
  ------------------------------------------------------------------------

  -- Levels 1 .. c_split, combinational off the accepted beat.
  reduce_front : process(all)
    variable v_sum : sum_tree_t;
    variable v_max : max_tree_t;
  begin
    for i in 0 to c_max_taps - 1 loop
      if tap_mask_q(i) = '1' then
        v_sum(0)(i) := resize(taps(i), c_sum_width);
        v_max(0)(i) := taps(i);
      else
        v_sum(0)(i) := (others => '0');
        v_max(0)(i) := to_signed(-128, 8);
      end if;
    end loop;

    for level in 1 to c_split1 loop
      for i in 0 to level_count(level) - 1 loop
        if 2 * i + 1 < level_count(level - 1) then
          v_sum(level)(i) := v_sum(level - 1)(2 * i) + v_sum(level - 1)(2 * i + 1);
          if v_max(level - 1)(2 * i) > v_max(level - 1)(2 * i + 1) then
            v_max(level)(i) := v_max(level - 1)(2 * i);
          else
            v_max(level)(i) := v_max(level - 1)(2 * i + 1);
          end if;
        else
          v_sum(level)(i) := v_sum(level - 1)(2 * i);
          v_max(level)(i) := v_max(level - 1)(2 * i);
        end if;
      end loop;
    end loop;

    front_max <= v_max(c_split1);
    front_sum <= v_sum(c_split1);
  end process;

  -- Levels c_split1+1 .. c_split2, combinational off stage 1's registers.
  reduce_mid : process(all)
    variable v_sum : sum_tree_t;
    variable v_max : max_tree_t;
  begin
    v_sum(c_split1) := p1_sum_q;
    v_max(c_split1) := p1_max_q;

    for level in c_split1 + 1 to c_split2 loop
      for i in 0 to level_count(level) - 1 loop
        if 2 * i + 1 < level_count(level - 1) then
          v_sum(level)(i) := v_sum(level - 1)(2 * i) + v_sum(level - 1)(2 * i + 1);
          if v_max(level - 1)(2 * i) > v_max(level - 1)(2 * i + 1) then
            v_max(level)(i) := v_max(level - 1)(2 * i);
          else
            v_max(level)(i) := v_max(level - 1)(2 * i + 1);
          end if;
        else
          v_sum(level)(i) := v_sum(level - 1)(2 * i);
          v_max(level)(i) := v_max(level - 1)(2 * i);
        end if;
      end loop;
    end loop;

    mid_max <= v_max(c_split2);
    mid_sum <= v_sum(c_split2);
  end process;

  -- Levels c_split2+1 .. c_levels, combinational off stage 2's registers.
  reduce_back : process(all)
    variable v_sum : sum_tree_t;
    variable v_max : max_tree_t;
  begin
    v_sum(c_split2) := p2_sum_q;
    v_max(c_split2) := p2_max_q;

    for level in c_split2 + 1 to c_levels loop
      for i in 0 to level_count(level) - 1 loop
        if 2 * i + 1 < level_count(level - 1) then
          v_sum(level)(i) := v_sum(level - 1)(2 * i) + v_sum(level - 1)(2 * i + 1);
          if v_max(level - 1)(2 * i) > v_max(level - 1)(2 * i + 1) then
            v_max(level)(i) := v_max(level - 1)(2 * i);
          else
            v_max(level)(i) := v_max(level - 1)(2 * i + 1);
          end if;
        else
          v_sum(level)(i) := v_sum(level - 1)(2 * i);
          v_max(level)(i) := v_max(level - 1)(2 * i);
        end if;
      end loop;
    end loop;

    max_result <= v_max(c_levels)(0);
    avgsum_result <= resize(v_sum(c_levels)(0), g_accum_width);
  end process;

  ------------------------------------------------------------------------
  -- Elastic control.
  --
  -- 'ready' is a function of this entity's own registers only: a beat may
  -- enter stage 1 whenever stage 1 is empty, or whenever stage 1 can move
  -- on because the output buffer is not full. It deliberately ignores the
  -- fact that a pop may free a slot in the same cycle -- taking credit for
  -- that would put the downstream 'ready' back into this signal, which is
  -- the whole thing the two-entry buffer exists to avoid. The buffer
  -- settles at one entry under a ready sink, so nothing is lost: one beat
  -- in and one beat out per cycle.
  ------------------------------------------------------------------------

  out_valid <= '1' when buf_count_q > 0 else '0';
  selected_output_ready <= m_avgsum_s2m.ready when out_is_avg_q(0) = '1' else m_max_s2m.ready;

  pop <= out_valid and selected_output_ready;
  -- Stage 2 -> output buffer, and stage 1 -> stage 2. Both deliberately
  -- ignore the fact that a pop may free a buffer slot in the same cycle,
  -- for the same reason the single-stage version did: taking credit for
  -- it would put the downstream 'ready' back into 's_window_s2m.ready'.
  push <= p2_valid_q when buf_count_q < c_buf_depth else '0';
  push1 <= p1_valid_q when (p2_valid_q = '0' or push = '1') else '0';

  s_window_s2m.ready <= cfg_match and ((not p1_valid_q) or push1);
  accepted <= s_window_m2s.valid and s_window_s2m.ready;

  pipeline : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        p1_valid_q <= '0';
        p2_valid_q <= '0';
        buf_count_q <= 0;
      else
        -- Stage 1. Held (not overwritten) whenever it cannot move on.
        if p1_valid_q = '0' or push1 = '1' then
          p1_valid_q <= accepted;
          if accepted = '1' then
            p1_is_avg_q <= is_avg_sel;
            p1_last_q <= s_window_m2s.last;
            p1_max_q <= front_max;
            p1_sum_q <= front_sum;
          end if;
        end if;

        -- Stage 2, same shape: held whenever it cannot move on.
        if p2_valid_q = '0' or push = '1' then
          p2_valid_q <= push1;
          if push1 = '1' then
            p2_is_avg_q <= p1_is_avg_q;
            p2_last_q <= p1_last_q;
            p2_max_q <= mid_max;
            p2_sum_q <= mid_sum;
          end if;
        end if;

        -- Output buffer: shift down on a pop, append on a push.
        if pop = '1' then
          for e in 0 to c_buf_depth - 2 loop
            out_is_avg_q(e) <= out_is_avg_q(e + 1);
            out_last_q(e) <= out_last_q(e + 1);
            out_max_q(e) <= out_max_q(e + 1);
            out_avgsum_q(e) <= out_avgsum_q(e + 1);
          end loop;
        end if;

        if push = '1' then
          for e in 0 to c_buf_depth - 1 loop
            -- The slot the pushed beat lands in: the tail, one lower if a
            -- pop is shifting the queue down in this same cycle.
            if e = buf_count_q - to_integer_sl(pop) then
              out_is_avg_q(e) <= p2_is_avg_q;
              out_last_q(e) <= p2_last_q;
              out_max_q(e) <= max_result;
              out_avgsum_q(e) <= avgsum_result;
            end if;
          end loop;
        end if;

        if push = '1' and pop = '0' then
          buf_count_q <= buf_count_q + 1;
        elsif push = '0' and pop = '1' then
          buf_count_q <= buf_count_q - 1;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------
  -- Output routing: mutually exclusive by construction (one buffer entry,
  -- one tag bit per entry) -- never both 'valid' in the same cycle.
  ------------------------------------------------------------------------

  m_max_m2s.valid <= out_valid and not out_is_avg_q(0);
  m_max_m2s.last <= out_last_q(0);
  m_max_m2s.data <=
    std_ulogic_vector(to_unsigned(0, axi_stream_data_sz - 8)) & std_ulogic_vector(out_max_q(0));
  m_max_m2s.user <= (others => '-');

  m_avgsum_m2s.valid <= out_valid and out_is_avg_q(0);
  m_avgsum_m2s.last <= out_last_q(0);
  m_avgsum_m2s.data <=
    std_ulogic_vector(to_unsigned(0, axi_stream_data_sz - g_accum_width)) &
    std_ulogic_vector(out_avgsum_q(0));
  m_avgsum_m2s.user <= (others => '-');

end architecture a;
