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

  -- Small, directed generics -- see proposal doc section 10.
  constant c_depth : positive := 4;
  constant c_pe_rows : positive := 2;
  constant c_pe_cols : positive := 2;
  constant c_accum_width : positive := 16;

  constant c_weight_lanes : positive := c_pe_rows * c_pe_cols;
  constant c_bias_lanes : positive := c_pe_rows;

  constant c_addr_width : positive := num_bits_needed(c_depth - 1);

  constant c_clk_period : time := 10 ns;
  constant c_settle : time := 1 ns;

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '0';

  signal s_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_stream_s2m : axi_stream_s2m_t;

  signal fill_bank_sel : std_ulogic := '0';
  signal fill_is_bias : std_ulogic := '0';
  signal read_bank_sel : std_ulogic := '0';

  signal weight_rd_addr : std_ulogic_vector(c_addr_width - 1 downto 0) := (others => '0');
  signal weight_rd_data : std_ulogic_vector(8 * c_weight_lanes - 1 downto 0);

  signal bias_rd_addr : std_ulogic_vector(c_addr_width - 1 downto 0) := (others => '0');
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

  -- Select a new fill bank: this is the "start a new fill session" edge
  -- the DUT detects on 'fill_bank_sel'. One idle clock keeps the edge
  -- unambiguous with respect to any in-flight 'valid'.
  procedure select_fill_bank(signal clk_i : in std_ulogic; signal sel : out std_ulogic; value : in std_ulogic) is
  begin
    sel <= value;
    wait until rising_edge(clk_i);
    wait for c_settle;
  end procedure;

begin

  clk <= not clk after c_clk_period / 2;

  dut : entity cnn_accel.cnn_accel_weight_buffer
    generic map (
      g_weight_buffer_depth => c_depth,
      g_pe_rows => c_pe_rows,
      g_pe_cols => c_pe_cols,
      g_accum_width => c_accum_width
    )
    port map (
      clk => clk,
      reset => reset,
      s_stream_m2s => s_stream_m2s,
      s_stream_s2m => s_stream_s2m,
      fill_bank_sel => fill_bank_sel,
      fill_is_bias => fill_is_bias,
      read_bank_sel => read_bank_sel,
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

    -- Fill a whole bank (weight region then bias region) with a
    -- 'salt'-tagged pattern. Assumes a fresh fill session was just
    -- started for 'bank' via 'select_fill_bank'.
    procedure fill_whole_bank(bank : std_ulogic; salt : natural) is
    begin
      select_fill_bank(clk, fill_bank_sel, bank);
      fill_is_bias <= '0';
      for r in 0 to c_depth - 1 loop
        push_weight_row(clk, s_stream_m2s, s_stream_s2m, r, salt);
      end loop;
      fill_is_bias <= '1';
      for r in 0 to c_depth - 1 loop
        push_bias_row(clk, s_stream_m2s, s_stream_s2m, r, salt);
      end loop;
    end procedure;

    procedure check_weight_row(bank : std_ulogic; row : natural; expected : std_ulogic_vector; msg : string) is
    begin
      read_bank_sel <= bank;
      weight_rd_addr <= std_ulogic_vector(to_unsigned(row, c_addr_width));
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(weight_rd_data, expected, msg);
    end procedure;

    procedure check_bias_row(bank : std_ulogic; row : natural; expected : std_ulogic_vector; msg : string) is
    begin
      read_bank_sel <= bank;
      bias_rd_addr <= std_ulogic_vector(to_unsigned(row, c_addr_width));
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(bias_rd_data, expected, msg);
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);

    do_reset;

    if run("test_ping_pong_fill_a_read_b") then
      -- Preload bank B fully, then start filling bank A while
      -- continuously reading bank B's already-written rows -- the fill
      -- (A) and read (B) ports are exercised on the very same clock
      -- edges, proving independence of the two banks/ports.
      fill_whole_bank('1', 200);

      select_fill_bank(clk, fill_bank_sel, '0');
      read_bank_sel <= '1';
      fill_is_bias <= '0';
      for r in 0 to c_depth - 1 loop
        for lane in 0 to c_weight_lanes - 1 loop
          weight_rd_addr <= std_ulogic_vector(to_unsigned(r mod c_depth, c_addr_width));
          push_fill_beat(clk, s_stream_m2s, s_stream_s2m,
            std_ulogic_vector(to_unsigned((r * 16 + lane) mod 256, 8)));
          wait for c_settle;
          check_equal(
            weight_rd_data, weight_row_expected(r mod c_depth, 200),
            "bank B weight row readback while filling bank A"
          );
        end loop;
      end loop;

      fill_is_bias <= '1';
      for r in 0 to c_depth - 1 loop
        for lane in 0 to c_bias_lanes - 1 loop
          bias_rd_addr <= std_ulogic_vector(to_unsigned(r mod c_depth, c_addr_width));
          push_fill_beat(clk, s_stream_m2s, s_stream_s2m,
            std_ulogic_vector(to_unsigned((r * 1000 + lane * 7) mod (2 ** c_accum_width), c_accum_width)));
          wait for c_settle;
          check_equal(
            bias_rd_data, bias_row_expected(r mod c_depth, 200),
            "bank B bias row readback while filling bank A"
          );
        end loop;
      end loop;

      -- And bank A itself now holds what was just streamed into it.
      for r in 0 to c_depth - 1 loop
        check_weight_row('0', r, weight_row_expected(r, 0), "bank A weight row after fill");
        check_bias_row('0', r, bias_row_expected(r, 0), "bank A bias row after fill");
      end loop;

    elsif run("test_ping_pong_fill_b_read_a") then
      -- Symmetric: preload bank A, then fill bank B while reading bank A.
      fill_whole_bank('0', 50);

      select_fill_bank(clk, fill_bank_sel, '1');
      read_bank_sel <= '0';
      fill_is_bias <= '0';
      for r in 0 to c_depth - 1 loop
        for lane in 0 to c_weight_lanes - 1 loop
          weight_rd_addr <= std_ulogic_vector(to_unsigned(r mod c_depth, c_addr_width));
          push_fill_beat(clk, s_stream_m2s, s_stream_s2m,
            std_ulogic_vector(to_unsigned((r * 16 + lane) mod 256, 8)));
          wait for c_settle;
          check_equal(
            weight_rd_data, weight_row_expected(r mod c_depth, 50),
            "bank A weight row readback while filling bank B"
          );
        end loop;
      end loop;

      for r in 0 to c_depth - 1 loop
        check_weight_row('1', r, weight_row_expected(r, 0), "bank B weight row after fill");
      end loop;

    elsif run("test_fill_pointer_autoincrement_and_reset_on_new_fill") then
      -- Fill only 1 of c_depth rows into bank A, verify only row 0
      -- reflects the new data; then start a *new* fill session for A
      -- (bank-select edge onto A again) and confirm it restarts at row 0,
      -- not row 1 -- the pointer-reset-on-new-fill behavior.
      select_fill_bank(clk, fill_bank_sel, '0');
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 7);
      check_weight_row('0', 0, weight_row_expected(0, 7), "row 0 after first partial fill beat");

      -- Force a new-fill-session edge on bank A: deselect then reselect.
      select_fill_bank(clk, fill_bank_sel, '1');
      select_fill_bank(clk, fill_bank_sel, '0');
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 99);
      check_weight_row(
        '0', 0, weight_row_expected(0, 99),
        "new fill session restarts bank A's weight row pointer at 0"
      );

    elsif run("test_bias_vs_weight_region_routing") then
      -- Within one fill session, weight beats and bias beats must land
      -- in their own region, never cross-contaminating the other.
      select_fill_bank(clk, fill_bank_sel, '0');
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 11);
      fill_is_bias <= '1';
      push_bias_row(clk, s_stream_m2s, s_stream_s2m, 0, 11);

      check_weight_row('0', 0, weight_row_expected(0, 11), "weight region holds weight-routed data");
      check_bias_row('0', 0, bias_row_expected(0, 11), "bias region holds bias-routed data");

    elsif run("test_read_latency_one_cycle") then
      -- Preload a known row, then confirm data is *not* valid the same
      -- cycle the address is presented, and *is* valid exactly one
      -- cycle later.
      fill_whole_bank('0', 3);

      read_bank_sel <= '0';
      weight_rd_addr <= std_ulogic_vector(to_unsigned(2, c_addr_width));
      wait until rising_edge(clk);
      wait for c_settle;
      -- One cycle after presenting the address: data must already match.
      check_equal(weight_rd_data, weight_row_expected(2, 3), "weight read data valid exactly 1 cycle after address");

      -- Change the address; on this same next cycle (before the next
      -- edge sees it), the *previous* row's data must still be held.
      weight_rd_addr <= std_ulogic_vector(to_unsigned(1, c_addr_width));
      -- Sampled just before the next rising edge: still the old (row 2) data.
      wait for c_clk_period / 2 - 2 * c_settle;
      check_equal(weight_rd_data, weight_row_expected(2, 3), "weight read data holds until the next clock edge");
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(weight_rd_data, weight_row_expected(1, 3), "weight read data updates 1 cycle after the new address");

      bias_rd_addr <= std_ulogic_vector(to_unsigned(3, c_addr_width));
      wait until rising_edge(clk);
      wait for c_settle;
      check_equal(bias_rd_data, bias_row_expected(3, 3), "bias read data valid exactly 1 cycle after address");

    elsif run("test_backpressure_weight_region_full") then
      -- Fill the weight region of bank A completely (c_depth rows);
      -- ready must deassert exactly once that region's row pointer
      -- reaches g_weight_buffer_depth, and must stay deasserted for
      -- further weight beats while the bias region (independent
      -- pointer) is unaffected.
      select_fill_bank(clk, fill_bank_sel, '0');
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
      check_weight_row('0', 0, weight_row_expected(0, 0), "row 0 unchanged: beat while full must not be accepted");

      -- The bias region's own pointer is untouched by the weight region
      -- being full: bias fills must still proceed normally.
      fill_is_bias <= '1';
      wait for c_settle;
      check_true(s_stream_s2m.ready = '1', "bias region ready is independent of a full weight region");
      push_bias_row(clk, s_stream_m2s, s_stream_s2m, 0, 0);
      check_bias_row('0', 0, bias_row_expected(0, 0), "bias region still fillable while weight region is full");

    elsif run("test_backpressure_bias_region_full") then
      -- Symmetric: fill the bias region completely, verify ready
      -- deasserts, while the (still-empty) weight region stays ready.
      select_fill_bank(clk, fill_bank_sel, '1');
      fill_is_bias <= '1';
      for r in 0 to c_depth - 1 loop
        check_true(s_stream_s2m.ready = '1', "ready before bias region is full");
        push_bias_row(clk, s_stream_m2s, s_stream_s2m, r, 0);
      end loop;

      wait for c_settle;
      check_true(s_stream_s2m.ready = '0', "ready deasserts once the bias region write pointer reaches depth");

      fill_is_bias <= '0';
      wait for c_settle;
      check_true(s_stream_s2m.ready = '1', "weight region ready is independent of a full bias region");

    elsif run("test_reset_mid_fill_does_not_leak_partial_bank") then
      -- Partially fill bank A (1 of c_depth rows) with pattern P1, abort
      -- via 'reset' mid-fill, then run a *full* new fill of bank A with
      -- pattern P2. If the row pointer had not truly reset to 0, row 0
      -- would still show P1 once the new fill wrote rows starting at 1;
      -- every row (including row 0) must show P2.
      select_fill_bank(clk, fill_bank_sel, '0');
      fill_is_bias <= '0';
      push_weight_row(clk, s_stream_m2s, s_stream_s2m, 0, 1);

      do_reset;

      select_fill_bank(clk, fill_bank_sel, '0');
      fill_is_bias <= '0';
      for r in 0 to c_depth - 1 loop
        push_weight_row(clk, s_stream_m2s, s_stream_s2m, r, 2);
      end loop;

      for r in 0 to c_depth - 1 loop
        check_weight_row(
          '0', r, weight_row_expected(r, 2),
          "post-abort fill fully overwrites bank A, no leftover partial-fill row"
        );
      end loop;

      -- And the freshly-completed bank now correctly reports full
      -- (ready deasserted), not still "partially filled".
      wait for c_settle;
      check_true(s_stream_s2m.ready = '0', "bank A weight region correctly full after the post-abort re-fill");
    end if;

    test_runner_cleanup(runner);
    wait;
  end process;

  test_runner_watchdog(runner, 1 ms);

end architecture tb;
