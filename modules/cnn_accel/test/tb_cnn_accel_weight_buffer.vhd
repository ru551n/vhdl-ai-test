library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;
use math.math_pkg.all;

library cnn_accel;

-- VUnit-5 testbench for cnn_accel_weight_buffer. See
-- modules/cnn_accel/doc/cnn_accel_weight_buffer_req.md and
-- modules/cnn_accel/doc/cnn_accel_weight_buffer_proposal.md section 10
-- for the verification plan.
entity tb_cnn_accel_weight_buffer is
  generic (runner_cfg : string);
end entity tb_cnn_accel_weight_buffer;

architecture tb of tb_cnn_accel_weight_buffer is

  -- Small, directed generics -- see proposal doc section 10. Weight and
  -- bias depths are deliberately different (4 vs 2) to exercise the
  -- decoupled 'g_weight_buffer_depth'/'g_bias_buffer_depth' address
  -- widths. The prefetch FIFO is disabled ('g_fill_fifo_depth => 0',
  -- straight-through) so every backpressure/latency check below sees the
  -- exact same-cycle timing as the region memories themselves -- the FIFO
  -- datapath itself is exercised by the default-generic instance in
  -- cnn_accel_conv_core / the weight_buffer netlist build.
  constant c_depth : positive := 4;
  constant c_bias_depth : positive := 2;
  constant c_pe_rows : positive := 2;
  constant c_pe_cols : positive := 2;
  constant c_accum_width : positive := 16;

  constant c_weight_lanes : positive := c_pe_rows * c_pe_cols;
  constant c_bias_lanes : positive := c_pe_rows;

  constant c_addr_width : positive := num_bits_needed(c_depth - 1);
  constant c_bias_addr_width : positive := num_bits_needed(c_bias_depth - 1);

  constant c_clk_period : time := 10 ns;
  constant c_settle : time := 1 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '0';

  signal s_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_stream_s2m : axi_stream_s2m_t;

  signal fill_start : std_ulogic := '0';
  signal fill_is_bias : std_ulogic := '0';

  signal weight_rd_addr : std_ulogic_vector(c_addr_width - 1 downto 0) := (others => '0');
  signal weight_rd_data : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0);

  signal bias_rd_addr : std_ulogic_vector(c_bias_addr_width - 1 downto 0) := (others => '0');
  signal bias_rd_data : std_ulogic_vector(c_accum_width * c_bias_lanes - 1 downto 0);

  ------------------------------------------------------------------------
  -- Deterministic per-row/lane test patterns. Kept in sync by construction
  -- with the 'push_weight_row'/'push_bias_row' procedures below.
  ------------------------------------------------------------------------

  function weight_row_expected(row : natural; salt : natural) return std_ulogic_vector is
    variable result : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0);
  begin
    for lane in 0 to c_weight_lanes - 1 loop
      result(8 * (lane + 1) - 1 downto 8 * lane) :=
        std_ulogic_vector(to_unsigned((salt + row * 16 + lane) mod 256, 8));
    end loop;
    return result;
  end function;

  function bias_row_expected(row : natural; salt : natural) return std_ulogic_vector is
    variable result : std_ulogic_vector(c_accum_width * c_bias_lanes - 1 downto 0);
  begin
    for lane in 0 to c_bias_lanes - 1 loop
      result(c_accum_width * (lane + 1) - 1 downto c_accum_width * lane) :=
        std_ulogic_vector(to_unsigned((salt + row * 1000 + lane * 7) mod (2 ** c_accum_width), c_accum_width));
    end loop;
    return result;
  end function;

  ------------------------------------------------------------------------
  -- Fill-side helper: push one lane onto the AXI4-Stream fill port,
  -- honoring 'ready' backpressure. No VUnit VC used -- see proposal doc
  -- section 10.
  ------------------------------------------------------------------------

  procedure push_fill_beat(
    signal clk_i : in std_ulogic;
    signal m2s : out axi_stream_m2s_t;
    signal s2m : in axi_stream_s2m_t;
    data_value : in std_ulogic_vector
  ) is
  begin
    m2s.data <= (others => '0');
    m2s.data(data_value'length - 1 downto 0) <= data_value;
    m2s.valid <= '1';
    wait until rising_edge(clk_i) and s2m.ready = '1';
    m2s.valid <= '0';
  end procedure;

  procedure push_weight_row(
    signal clk_i : in std_ulogic;
    signal m2s : out axi_stream_m2s_t;
    signal s2m : in axi_stream_s2m_t;
    row : in natural;
    salt : in natural
  ) is
  begin
    for lane in 0 to c_weight_lanes - 1 loop
      push_fill_beat(
        clk_i, m2s, s2m,
        std_ulogic_vector(to_unsigned((salt + row * 16 + lane) mod 256, 8))
      );
    end loop;
  end procedure;

  procedure push_bias_row(
    signal clk_i : in std_ulogic;
    signal m2s : out axi_stream_m2s_t;
    signal s2m : in axi_stream_s2m_t;
    row : in natural;
    salt : in natural
  ) is
  begin
    for lane in 0 to c_bias_lanes - 1 loop
      push_fill_beat(
        clk_i, m2s, s2m,
        std_ulogic_vector(to_unsigned((salt + row * 1000 + lane * 7) mod (2 ** c_accum_width), c_accum_width))
      );
    end loop;
  end procedure;

  -- Pulse 'fill_start' for exactly one cycle: begins a new fill session
  -- (both regions' write pointers/lane indices/row-assembly registers
  -- reset to 0 together). One settle delay before returning keeps the
  -- pulse unambiguous with respect to whatever the caller drives next.
  procedure pulse_fill_start(signal clk_i : in std_ulogic; signal fs : out std_ulogic) is
  begin
    fs <= '1';
    wait until rising_edge(clk_i);
    wait for c_settle;
    fs <= '0';
  end procedure;

begin

  clk <= not clk after c_clk_period / 2;

  dut : entity cnn_accel.cnn_accel_weight_buffer
    generic map (
      g_weight_buffer_depth => c_depth,
      g_bias_buffer_depth => c_bias_depth,
      g_pe_rows => c_pe_rows,
      g_pe_cols => c_pe_cols,
      g_accum_width => c_accum_width,
      g_fill_fifo_depth => 0
    )
    port map (
      clk => clk,
      reset => reset,
      s_stream_m2s => s_stream_m2s,
      s_stream_s2m => s_stream_s2m,
      fill_start => fill_start,
      fill_is_bias => fill_is_bias,
      weight_rd_addr => weight_rd_addr,
      weight_rd_data => weight_rd_data,
      bias_rd_addr => bias_rd_addr,
      bias_rd_data => bias_rd_data
    );

  ------------------------------------------------------------------------
  main : process
    variable row : natural;

    procedure do_reset is
    begin
      reset <= '1';
      wait until rising_edge(clk);
      wait for c_settle;
      reset <= '0';
    end procedure;

    -- Fill the whole weight set (weight region then bias region) with a
    -- 'salt'-tagged pattern. Pulses 'fill_start' itself, so the caller
    -- just supplies the pattern.
    procedure fill_whole_buffer(salt : natural) is
    begin
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      for r in 0 to c_depth - 1 loop
        push_weight_row(clk, s_stream_m2s, s_stream_s2m, r, salt);
      end loop;
      fill_is_bias <= '1';
      for r in 0 to c_bias_depth - 1 loop
        push_bias_row(clk, s_stream_m2s, s_stream_s2m, r, salt);
      end loop;
    end procedure;

    -- The weight region reads through the block RAM's output register, so
    -- data is valid TWO cycles after the address (see the DUT's read
    -- process). The bias/scale regions are still one.
    procedure check_weight_row(row : natural; expected : std_ulogic_vector; msg : string) is
    begin
      weight_rd_addr <= std_ulogic_vector(to_unsigned(row, c_addr_width));
      wait until rising_edge(clk);
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(weight_rd_data, expected, msg);
    end procedure;

    procedure check_bias_row(row : natural; expected : std_ulogic_vector; msg : string) is
    begin
      bias_rd_addr <= std_ulogic_vector(to_unsigned(row, c_bias_addr_width));
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(bias_rd_data, expected, msg);
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    do_reset;

    if run("test_fill_then_read_weight_and_bias") then
      -- Fill the whole (single-buffered) weight set and read every row of
      -- both regions back.
      fill_whole_buffer(5);

      for r in 0 to c_depth - 1 loop
        check_weight_row(r, weight_row_expected(r, 5), "weight row after fill");
      end loop;
      for r in 0 to c_bias_depth - 1 loop
        check_bias_row(r, bias_row_expected(r, 5), "bias row after fill");
      end loop;

    elsif run("test_fill_start_restarts_pointer_and_drops_same_cycle_beat") then
      -- Fill row 0 fully with pattern P1.
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 7);
      check_weight_row(0, weight_row_expected(0, 7), "row 0 after first fill session");

      -- A new 'fill_start' pulse must restart the write pointer at row 0,
      -- not continue at row 1 -- refill with pattern P2 and confirm row 0
      -- shows P2, not P1.
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 99);
      check_weight_row(0, weight_row_expected(0, 99), "new fill session restarts the weight row pointer at 0");

      -- A beat presented on the very same cycle as 'fill_start' must not
      -- be accepted, even though 'ready' is asserted that cycle (the
      -- region is far from full): drive a poison beat concurrently with
      -- the pulse, then confirm the next real fill (pattern P3) is what
      -- row 0 ends up holding.
      s_stream_m2s.data <= (others => '0');
      s_stream_m2s.data(7 downto 0) <= x"AA";
      s_stream_m2s.valid <= '1';
      fill_start <= '1';
      wait until rising_edge(clk);
      wait for c_settle;
      fill_start <= '0';
      s_stream_m2s.valid <= '0';

      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 42);
      check_weight_row(
        0, weight_row_expected(0, 42),
        "same-cycle-as-fill_start beat dropped: row 0 holds only the real post-pulse fill"
      );

    elsif run("test_bias_vs_weight_region_routing") then
      -- Within one fill session, weight beats and bias beats must land
      -- in their own region, never cross-contaminating the other.
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 11);
      fill_is_bias <= '1';
      push_bias_row(clk, s_stream_m2s, s_stream_s2m, 0, 11);

      check_weight_row(0, weight_row_expected(0, 11), "weight region holds weight-routed data");
      check_bias_row(0, bias_row_expected(0, 11), "bias region holds bias-routed data");

    elsif run("test_read_latency") then
      -- Preload a known pattern, then pin down BOTH regions' read
      -- latency exactly: the weight region reads through the block RAM's
      -- output register and is valid two cycles after the address (never
      -- one, which is checked explicitly below), the bias region is
      -- valid after one. Renamed from 'test_read_latency_one_cycle' when
      -- the weight region gained that output register -- see the DUT's
      -- read process for why it did.
      fill_whole_buffer(3);

      -- 'weight_rd_addr' has been 0 since reset, so row 0 is what both
      -- read stages hold going in; that is what makes the "not yet"
      -- check below a real check and not a check against 'X'.
      weight_rd_addr <= std_ulogic_vector(to_unsigned(2, c_addr_width));
      wait until rising_edge(clk);
      wait for c_settle;
      -- One cycle after presenting the address: row 2 has only reached
      -- the RAM's DO stage, so the output register still presents row 0.
      check_equal(
        weight_rd_data, weight_row_expected(0, 3),
        "weight read data is NOT valid 1 cycle after the address"
      );
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(
        weight_rd_data, weight_row_expected(2, 3),
        "weight read data valid exactly 2 cycles after address"
      );

      -- Change the address; on this same next cycle (before the next
      -- edge sees it), the *previous* row's data must still be held.
      weight_rd_addr <= std_ulogic_vector(to_unsigned(1, c_addr_width));
      -- Sampled just before the next rising edge: still the old (row 2) data.
      wait for c_clk_period / 2 - 2 * c_settle;
      check_equal(weight_rd_data, weight_row_expected(2, 3), "weight read data holds until the next clock edge");
      wait until rising_edge(clk);
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(
        weight_rd_data, weight_row_expected(1, 3),
        "weight read data updates 2 cycles after the new address"
      );

      bias_rd_addr <= std_ulogic_vector(to_unsigned(c_bias_depth - 1, c_bias_addr_width));
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(
        bias_rd_data, bias_row_expected(c_bias_depth - 1, 3),
        "bias read data valid exactly 1 cycle after address"
      );

    elsif run("test_backpressure_weight_region_full") then
      -- Fill the weight region completely (c_depth rows); ready must
      -- deassert exactly once that region's row pointer reaches
      -- g_weight_buffer_depth, and must stay deasserted for further
      -- weight beats while the bias region (independent pointer) is
      -- unaffected.
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      for r in 0 to c_depth - 1 loop
        check_true(s_stream_s2m.ready = '1', "ready before weight region is full");
        push_weight_row(clk, s_stream_m2s, s_stream_s2m, r, 0);
      end loop;

      wait for c_settle;
      check_true(s_stream_s2m.ready = '0', "ready deasserts once the weight region write pointer reaches depth");

      -- Present another weight beat: must not be accepted (row 0 must
      -- keep its original value, not get overwritten).
      s_stream_m2s.data <= (others => '0');
      s_stream_m2s.data(7 downto 0) <= x"AA";
      s_stream_m2s.valid <= '1';
      wait until rising_edge(clk);
      wait for c_settle;
      s_stream_m2s.valid <= '0';
      check_weight_row(0, weight_row_expected(0, 0), "row 0 unchanged: beat while full must not be accepted");

      -- The bias region's own pointer is untouched by the weight region
      -- being full: bias fills must still proceed normally.
      fill_is_bias <= '1';
      wait for c_settle;
      check_true(s_stream_s2m.ready = '1', "bias region ready is independent of a full weight region");
      push_bias_row(clk, s_stream_m2s, s_stream_s2m, 0, 0);
      check_bias_row(0, bias_row_expected(0, 0), "bias region still fillable while weight region is full");

    elsif run("test_backpressure_bias_region_full") then
      -- Symmetric: fill the bias region completely, verify ready
      -- deasserts, while the (still-empty) weight region stays ready.
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '1';
      for r in 0 to c_bias_depth - 1 loop
        check_true(s_stream_s2m.ready = '1', "ready before bias region is full");
        push_bias_row(clk, s_stream_m2s, s_stream_s2m, r, 0);
      end loop;

      wait for c_settle;
      check_true(s_stream_s2m.ready = '0', "ready deasserts once the bias region write pointer reaches depth");

      fill_is_bias <= '0';
      wait for c_settle;
      check_true(s_stream_s2m.ready = '1', "weight region ready is independent of a full bias region");

    elsif run("test_reset_mid_fill_does_not_leak_partial_fill") then
      -- Partially fill (1 of c_depth rows) with pattern P1, abort via
      -- 'reset' mid-fill, then run a *full* new fill with pattern P2. If
      -- the row pointer had not truly reset to 0, row 0 would still show
      -- P1 once the new fill wrote rows starting at 1; every row
      -- (including row 0) must show P2.
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 1);

      do_reset;

      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      for r in 0 to c_depth - 1 loop
        push_weight_row(clk, s_stream_m2s, s_stream_s2m, r, 2);
      end loop;

      for r in 0 to c_depth - 1 loop
        check_weight_row(
          r, weight_row_expected(r, 2),
          "post-abort fill fully overwrites the weight region, no leftover partial-fill row"
        );
      end loop;

      -- And the freshly-completed region now correctly reports full
      -- (ready deasserted), not still "partially filled".
      wait for c_settle;
      check_true(s_stream_s2m.ready = '0', "weight region correctly full after the post-abort re-fill");

    elsif run("test_read_already_committed_rows_during_active_fill") then
      -- Single-buffer means fill and read share the same memory: this
      -- proves the read port keeps returning the last *committed* row
      -- even while a different row's write is still mid-assembly (no
      -- second bank needed to keep row 0 stable while later rows fill).
      pulse_fill_start(clk, fill_start);
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 300);
      check_weight_row(0, weight_row_expected(0, 300), "row 0 committed before the concurrent-read check");

      for r in 1 to c_depth - 1 loop
        for lane in 0 to c_weight_lanes - 1 loop
          weight_rd_addr <= std_ulogic_vector(to_unsigned(0, c_addr_width));
          push_fill_beat(
            clk, s_stream_m2s, s_stream_s2m,
            std_ulogic_vector(to_unsigned((300 + r * 16 + lane) mod 256, 8))
          );
          wait for c_settle;
          check_equal(
            weight_rd_data, weight_row_expected(0, 300),
            "row 0 stays readable while a later row is still mid-assembly"
          );
        end loop;
      end loop;

      check_weight_row(
        c_depth - 1, weight_row_expected(c_depth - 1, 300),
        "last row fully committed after the concurrent fill/read pass"
      );
    end if;

    test_runner_cleanup(runner);
    wait;
  end process;

  test_runner_watchdog(runner, 1 ms);

end architecture tb;
