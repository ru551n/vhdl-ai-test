library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library fifo;

library math;
use math.math_pkg.all;

-- Generic AXI4-Stream 3x3 sliding-window generator over a continuous
-- raster-scan stream. See modules/canny/doc/canny_window3x3_req.md
-- and modules/canny/doc/canny_window3x3_proposal.md.
entity canny_window3x3 is
  generic (
    img_width  : positive;
    img_height : positive;
    data_width : positive;
    user_width : positive range 1 to 2
  );
  port (
    clk   : in std_logic;
    reset : in std_logic;

    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    s_axis_tdata  : in  std_logic_vector(data_width - 1 downto 0);
    s_axis_tuser  : in  std_logic_vector(user_width - 1 downto 0);
    s_axis_tlast  : in  std_logic;

    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    m_axis_tdata  : out std_logic_vector(9 * data_width - 1 downto 0);
    m_axis_tuser  : out std_logic_vector(1 downto 0);
    m_axis_tlast  : out std_logic
  );
end entity canny_window3x3;

architecture a of canny_window3x3 is

  -- Packed lane payload carried through the line-buffer FIFOs and the
  -- column-tap shift registers: tdata & border & sof (MSB to LSB). tlast is
  -- not carried as a passenger bit -- m_axis_tlast is recomputed from
  -- out_col instead (per requirement), so no such bit is needed here.
  constant lane_width : positive := data_width + 2;

  -- fifo.fifo_wrapper's synchronous-mode branch (fifo.vhd) requires a
  -- power-of-two memory depth; +1 gives headroom beyond img_width so the
  -- FIFO's own fullness signals are never load-bearing for correctness
  -- (this module's own accepted_beats counter decides read timing).
  constant fifo_depth : positive := round_up_to_power_of_two(img_width + 1);

  function pack_lane(
    tdata  : std_logic_vector(data_width - 1 downto 0);
    border : std_logic;
    sof    : std_logic
  ) return std_logic_vector is
  begin
    return tdata & border & sof;
  end function;

  function lane_data(lane : std_logic_vector(lane_width - 1 downto 0))
    return std_logic_vector is
  begin
    return lane(lane_width - 1 downto 2);
  end function;

  function lane_border(lane : std_logic_vector(lane_width - 1 downto 0))
    return std_logic is
  begin
    return lane(1);
  end function;

  function lane_sof(lane : std_logic_vector(lane_width - 1 downto 0))
    return std_logic is
  begin
    return lane(0);
  end function;

  -- True if the tap at (row_off, col_off) relative to the current output
  -- window center (out_row, out_col) lies within the frame.
  function tap_in_frame(
    out_row, out_col      : natural;
    row_off, col_off       : integer;
    img_width, img_height  : positive
  ) return boolean is
    variable trow, tcol : integer;
  begin
    trow := out_row + row_off;
    tcol := out_col + col_off;
    return trow >= 0 and trow <= img_height - 1 and tcol >= 0 and tcol <= img_width - 1;
  end function;

  signal fire : std_logic;
  signal down_ready : std_logic;

  signal border_in : std_logic;

  -- Bottom row (out_row + 1, the live/newest row): "right" is combinational
  -- (this cycle's live sample), "mid"/"left" are 1-/2-fire-cycle registered.
  signal bottom_live : std_logic_vector(lane_width - 1 downto 0);
  signal bottom_mid_reg, bottom_left_reg : std_logic_vector(lane_width - 1 downto 0)
    := (others => '0');

  -- Middle row (out_row, the window center row).
  signal mid_live : std_logic_vector(lane_width - 1 downto 0);
  signal mid_mid_reg, mid_left_reg : std_logic_vector(lane_width - 1 downto 0)
    := (others => '0');

  -- Top row (out_row - 1).
  signal top_live : std_logic_vector(lane_width - 1 downto 0);
  signal top_mid_reg, top_left_reg : std_logic_vector(lane_width - 1 downto 0)
    := (others => '0');

  signal fifo1_write_ready, fifo1_read_valid : std_logic;
  signal fifo1_read_data : std_logic_vector(lane_width - 1 downto 0);

  signal fifo2_write_ready, fifo2_read_valid : std_logic;
  signal fifo2_read_data : std_logic_vector(lane_width - 1 downto 0);

  -- Saturating counter of accepted (tvalid and tready) input beats, used to
  -- derive the priming state below. Comparisons against this monotonic,
  -- registered counter behave as sticky flags without needing separate
  -- flip-flops.
  signal accepted_beats : natural range 0 to 2 * img_width + 2 := 0;

  -- Saturating counters of total accepted input beats / total sent output
  -- beats (each up to a full frame). Used only to derive 'draining' below
  -- (the tail-flush condition) -- 'accepted_beats' above saturates far
  -- below a full frame's worth of beats for any image taller than a
  -- handful of rows, so it cannot itself be used to detect end-of-frame.
  signal input_beat_count : natural range 0 to img_width * img_height := 0;
  signal output_beat_count : natural range 0 to img_width * img_height := 0;

  signal mid_row_active, top_row_active : std_logic;

  -- 'primed': the fixed img_width + 1 accepted-beat priming threshold
  -- (see "Timing/latency" in doc/canny_window3x3.md). 'draining': all
  -- img_width * img_height input beats have been accepted but not
  -- every output window has been sent yet -- the tail-flush state needed
  -- because the last img_width + 1 output windows have no corresponding
  -- *new* input beat left to ride along with (see "Timing/latency").
  -- 'window_valid' (m_axis_tvalid) is high on a genuine fresh fire once
  -- primed, OR unconditionally while draining (draining implies no fresh
  -- fire is even possible, since all input has already been accepted).
  -- 'step' is the single shared "a real output beat is being consumed this
  -- cycle" pulse that both advances out_row/out_col and pulls the next
  -- buffered sample out of the mid/top row lanes (fifo reads + column-tap
  -- shifts), whether or not a fresh input fire is happening this cycle.
  signal primed, draining, window_valid, step : std_logic;
  signal all_input_received, all_output_sent : std_logic;

  signal out_row : natural range 0 to img_height - 1 := 0;
  signal out_col : natural range 0 to img_width - 1 := 0;

begin

  fire <= s_axis_tvalid and s_axis_tready;

  -- 'down_ready'/'s_axis_tready' must depend only on the sticky, registered
  -- 'primed'/'draining' state -- never on 'window_valid' itself, which
  -- below depends on 'fire' (and hence, transitively, on 'down_ready') --
  -- to avoid a combinational loop.
  down_ready <= (not (primed or draining)) or m_axis_tready;
  s_axis_tready <= down_ready and fifo1_write_ready
    and (fifo2_write_ready or not mid_row_active);

  -- Generate statements (not a conditional signal assignment) are required
  -- here: user_width=1 instances have s_axis_tuser'range = (0 downto 0),
  -- so a runtime "s_axis_tuser(1) when user_width = 2 else '0'" still
  -- elaborates/evaluates the out-of-bounds index and fails at simulation
  -- time even though that branch is never selected. A generate only
  -- elaborates the taken branch, so the out-of-range index never exists in
  -- the user_width=1 instance.
  gen_border_uw2 : if user_width = 2 generate
    border_in <= s_axis_tuser(1);
  end generate gen_border_uw2;
  gen_border_uw1 : if user_width /= 2 generate
    border_in <= '0';
  end generate gen_border_uw1;

  bottom_live <= pack_lane(s_axis_tdata, border_in, s_axis_tuser(0));
  mid_live    <= fifo1_read_data;
  top_live    <= fifo2_read_data;

  ------------------------------------------------------------------------------
  -- Priming thresholds, derived from the fixed (ungated) "bottom"/live-row
  -- register chain below: bottom_mid_reg/bottom_left_reg shift on every
  -- accepted beat starting from the very first one, so the raster position
  -- implied by those registers at any given accepted-beat count is fixed
  -- and cannot be delayed without literally discarding samples. Working
  -- back from that fixed timing (see doc/canny_window3x3.md "Timing/
  -- latency" for the full per-tap fencepost derivation):
  --   * mid_row_active turns on once row 0 has been fully written into
  --     fifo1, i.e. after img_width accepted beats -- the same instant
  --     row 1 (the first row needing a "middle" neighbor) starts arriving.
  --   * top_row_active mirrors that one row-buffer-depth later, after
  --     2*img_width accepted beats.
  --   * 'primed' (the fixed per-tap timing threshold) must turn on the
  --     instant the window centered at (out_row=0, out_col=0) is first
  --     assembled, which the per-tap timing forces to exactly
  --     img_width + 1 accepted beats (not 2*img_width + 2 -- that
  --     count is when the single most-delayed tap, top_left_reg, first
  --     holds non-garbage data, which is an internal detail with no
  --     bearing on 'primed': out-of-frame taps are zero-masked below
  --     regardless of whether their backing register happens to hold
  --     real or stale data yet).
  --
  -- IMPORTANT: 'primed' alone is NOT 'window_valid'/m_axis_tvalid -- see
  -- 'window_valid' below. Gating m_axis_tvalid directly off 'primed' (as
  -- an earlier, buggy revision of this module did) asserts m_axis_tvalid
  -- even on cycles with no fresh accepted input beat at all (e.g. any
  -- s_axis-side stall once primed), replaying the same stale window/
  -- out_row/out_col beat as a phantom extra output transfer whenever
  -- m_axis_tready happens to be high that same cycle -- a real,
  -- pervasive correctness bug (not just a tail/drain issue), since
  -- 'primed' is a sticky, monotonic flag that never deasserts once set.
  mid_row_active <= '1' when accepted_beats >= img_width else '0';
  top_row_active <= '1' when accepted_beats >= 2 * img_width else '0';
  primed         <= '1' when accepted_beats >= img_width + 1 else '0';

  -- Tail-flush ('draining') condition: once every img_width *
  -- img_height input beats has been accepted, no further *fresh* fire
  -- will ever happen for this frame, yet the last img_width + 1 output
  -- windows have not been produced yet (each of those windows' only
  -- missing ingredient -- the buffered mid/top row data -- is already
  -- sitting in fifo1/fifo2, it just has not been read out yet). Draining
  -- keeps 'window_valid' asserted (and 'step' pulsing on m_axis_tready)
  -- purely on downstream readiness, with no further dependency on
  -- s_axis_tvalid, until all img_width * img_height outputs have been
  -- sent. Without this, the last img_width + 1 output windows would
  -- never be produced at all, since out_row/out_col would otherwise only
  -- ever advance in lock-step with a fresh input fire.
  all_input_received <= '1' when input_beat_count >= img_width * img_height else '0';
  all_output_sent     <= '1' when output_beat_count >= img_width * img_height else '0';
  draining <= all_input_received and not all_output_sent;

  -- The actual m_axis_tvalid: fresh upstream data once primed, or
  -- unconditionally while draining (s_axis_tvalid is always '0' while
  -- draining -- all input has already been accepted -- so these two
  -- disjuncts never overlap).
  --
  -- IMPORTANT: this deliberately uses 's_axis_tvalid', NOT 'fire'
  -- (= s_axis_tvalid and s_axis_tready). 'down_ready'/s_axis_tready above
  -- already depends on m_axis_tready (once primed or draining), so using
  -- 'fire' here would make m_axis_tvalid transitively combinationally
  -- dependent on m_axis_tready -- a same-channel VALID-depends-on-READY
  -- combinational path that AXI4-Stream forbids (and that this module's
  -- own documentation, doc/canny_window3x3.md "Interfaces/protocols",
  -- explicitly calls out as a correctness requirement: "m_axis_tvalid is
  -- a function of internal state only ... never combinationally dependent
  -- on m_axis_tready"). Using 's_axis_tvalid' instead avoids that: while
  -- s_axis_tready happens to be '0' for any reason (e.g. m_axis_tready
  -- low, or transient FIFO backpressure), 'fire' is '0' so nothing
  -- advances (out_row/out_col and every tap register stay frozen), and
  -- m_axis_tvalid keeps re-offering that same still-unconsumed window --
  -- exactly the standard, correct elastic-pipeline stall behavior, not a
  -- phantom replay (a phantom replay is specifically a *fresh* fire
  -- happening while gated only by the sticky 'primed' flag with no
  -- accompanying real progress, which is what the pre-fix version of this
  -- module actually did -- see 'primed' above).
  window_valid <= (s_axis_tvalid and primed) or draining;

  -- The single shared "advance the mid/top row lanes (fifo reads + column-
  -- tap shifts) this cycle" pulse. This must track 'fire' directly (NOT
  -- 'window_valid'/'fire and primed'): mid_row_active/top_row_active turn
  -- on strictly *before* 'primed' does (see the priming thresholds above,
  -- img_width and 2*img_width vs. img_width + 1), so the mid/top
  -- row's fifo-read-and-shift pipeline must already be advancing on plain
  -- 'fire' during that pre-primed-but-row-active window, one beat ahead
  -- of 'window_valid' turning on -- gating it on 'window_valid' instead
  -- would silently drop that one beat's worth of fifo1/fifo2 reads,
  -- permanently desynchronizing the mid/top rows by one position for the
  -- rest of the frame. During draining (tail flush, no fresh fire left),
  -- the same pulse instead advances purely on m_axis_tready.
  step <= fire or (draining and m_axis_tready);

  ------------------------------------------------------------------------------
  priming_counter : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        accepted_beats <= 0;
        input_beat_count <= 0;
        output_beat_count <= 0;
      else
        if fire then
          if accepted_beats < 2 * img_width + 2 then
            accepted_beats <= accepted_beats + 1;
          end if;
          if input_beat_count < img_width * img_height then
            input_beat_count <= input_beat_count + 1;
          end if;
        end if;
        if window_valid and m_axis_tready then
          if output_beat_count < img_width * img_height then
            output_beat_count <= output_beat_count + 1;
          end if;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------------
  out_position_counter : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        out_row <= 0;
        out_col <= 0;
      elsif step and window_valid then
        if out_col = img_width - 1 then
          out_col <= 0;
          if out_row = img_height - 1 then
            out_row <= 0;
          else
            out_row <= out_row + 1;
          end if;
        else
          out_col <= out_col + 1;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------------
  -- The "bottom"/live row always shifts on a genuine fresh fire only (its
  -- 'right' tap is s_axis_tdata itself, so there is nothing to shift
  -- without one). The "mid"/"top" rows, however, must keep shifting off
  -- their own line-buffer FIFOs' buffered contents on 'step' (which also
  -- fires during draining, with no fresh input fire at all) -- otherwise
  -- the last img_width + 1 output windows would repeat stale mid/top
  -- data forever instead of advancing through the buffered tail of the
  -- frame.
  row_taps : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        bottom_mid_reg <= (others => '0');
        bottom_left_reg <= (others => '0');
        mid_mid_reg <= (others => '0');
        mid_left_reg <= (others => '0');
        top_mid_reg <= (others => '0');
        top_left_reg <= (others => '0');
      else
        if fire then
          bottom_left_reg <= bottom_mid_reg;
          bottom_mid_reg  <= bottom_live;
        end if;

        if step and mid_row_active then
          mid_left_reg <= mid_mid_reg;
          mid_mid_reg  <= mid_live;
        end if;

        if step and top_row_active then
          top_left_reg <= top_mid_reg;
          top_mid_reg  <= top_live;
        end if;
      end if;
    end if;
  end process;

  ------------------------------------------------------------------------------
  -- Line buffer 1: delays the live (bottom) row by img_width accepted
  -- beats to produce the middle (center) row.
  fifo1_inst : entity fifo.fifo_wrapper
    generic map (
      use_asynchronous_fifo => false,
      width => lane_width,
      depth => fifo_depth
    )
    port map (
      clk => clk,
      --
      write_ready => fifo1_write_ready,
      write_valid => fire,
      write_data => bottom_live,
      --
      read_ready => step and mid_row_active,
      read_valid => fifo1_read_valid,
      read_data => fifo1_read_data
    );

  ------------------------------------------------------------------------------
  -- Line buffer 2: delays the middle row by another img_width accepted
  -- beats to produce the top row.
  fifo2_inst : entity fifo.fifo_wrapper
    generic map (
      use_asynchronous_fifo => false,
      width => lane_width,
      depth => fifo_depth
    )
    port map (
      clk => clk,
      --
      write_ready => fifo2_write_ready,
      write_valid => fire and mid_row_active,
      write_data => mid_live,
      --
      read_ready => step and top_row_active,
      read_valid => fifo2_read_valid,
      read_data => fifo2_read_data
    );

  ------------------------------------------------------------------------------
  m_axis_tvalid <= window_valid;
  m_axis_tlast <= '1' when out_col = img_width - 1 else '0';
  m_axis_tuser(0) <= lane_sof(mid_mid_reg);

  ------------------------------------------------------------------------------
  -- NOTE: uses an explicit sensitivity list rather than 'process(all)'.
  -- Empirically (see waveform evidence gathered while debugging this
  -- module), GHDL 7.0.0-dev's 'all' sensitivity-list inference does not
  -- reliably pick up signals that are read only indirectly, through the
  -- 'tap_lane' impure function called from inside the nested for-loops
  -- below: with 'process(all)', m_axis_tdata was observed to be computed
  -- once (at elaboration/reset, using whatever the tap registers/lanes
  -- held at that instant) and then never recomputed for the rest of the
  -- simulation despite bottom_live/bottom_mid_reg/mid_live/mid_mid_reg/
  -- etc. genuinely changing every cycle -- so m_axis_tdata kept
  -- reporting stale/garbage (in one observed case, still-'U') data even
  -- on cycles where m_axis_tvalid was asserted and every input signal it
  -- should depend on already held the correct value. Listing every
  -- signal actually read by this process (directly or via tap_lane)
  -- explicitly avoids relying on that inference at all.
  assemble_window : process(
    out_row, out_col,
    bottom_live, bottom_mid_reg, bottom_left_reg,
    mid_live, mid_mid_reg, mid_left_reg,
    top_live, top_mid_reg, top_left_reg
  )
    type tap_row_t is (top_row, mid_row, bot_row);
    type tap_col_t is (left_col, mid_col, right_col);

    variable border_or : std_logic;

    impure function tap_lane(row : tap_row_t; col : tap_col_t)
      return std_logic_vector is
    begin
      case row is
        when top_row =>
          case col is
            when left_col  => return top_left_reg;
            when mid_col   => return top_mid_reg;
            when right_col => return top_live;
          end case;
        when mid_row =>
          case col is
            when left_col  => return mid_left_reg;
            when mid_col   => return mid_mid_reg;
            when right_col => return mid_live;
          end case;
        when bot_row =>
          case col is
            when left_col  => return bottom_left_reg;
            when mid_col   => return bottom_mid_reg;
            when right_col => return bottom_live;
          end case;
      end case;
    end function;

    function row_off(row : tap_row_t) return integer is
    begin
      case row is
        when top_row => return -1;
        when mid_row => return 0;
        when bot_row => return 1;
      end case;
    end function;

    function col_off(col : tap_col_t) return integer is
    begin
      case col is
        when left_col  => return -1;
        when mid_col   => return 0;
        when right_col => return 1;
      end case;
    end function;

    variable lane : std_logic_vector(lane_width - 1 downto 0);
    variable in_frame : boolean;
    variable edge_here : std_logic;
  begin
    border_or := '0';

    for row in tap_row_t'left to tap_row_t'right loop
      for col in tap_col_t'left to tap_col_t'right loop
        lane := tap_lane(row, col);

        in_frame := tap_in_frame(
          out_row, out_col, row_off(row), col_off(col), img_width, img_height
        );

        if in_frame then
          m_axis_tdata(
            (9 - (3 * tap_row_t'pos(row) + tap_col_t'pos(col))) * data_width - 1
            downto (8 - (3 * tap_row_t'pos(row) + tap_col_t'pos(col))) * data_width
          ) <= lane_data(lane);
          border_or := border_or or lane_border(lane);
        else
          m_axis_tdata(
            (9 - (3 * tap_row_t'pos(row) + tap_col_t'pos(col))) * data_width - 1
            downto (8 - (3 * tap_row_t'pos(row) + tap_col_t'pos(col))) * data_width
          ) <= (others => '0');
        end if;
      end loop;
    end loop;

    if out_row = 0 or out_row = img_height - 1
      or out_col = 0 or out_col = img_width - 1
    then
      edge_here := '1';
    else
      edge_here := '0';
    end if;

    m_axis_tuser(1) <= border_or or edge_here;
  end process;

end architecture a;
