library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.axi_stream_pkg.all;
use vunit_lib.sync_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

-- TDD testbench for modules/canny_window3x3/src/canny_window3x3.vhd, written
-- before the --@ markers in that file are resolved (per the project's TDD
-- policy in shared/Vunit.md #15). Uses VUnit's raw axi_stream_master/
-- axi_stream_slave verification components directly (not the hdl-modules
-- bfm.* wrappers), because bfm.axi_stream_master/slave require
-- user_width mod 8 = 0 and this module's s_axis_tuser can be 1 bit wide --
-- see shared/Vunit.md #12 and vhtestgen/SKILL.md.
--
-- Two independent DUT instances are used since g_user_width is a
-- generic (affects s_axis_tuser's port width, so it can not be swept at
-- run time on a single instance): dut_uw1 (g_user_width => 1, no incoming
-- border bit at all) is used for test_random_data/test_full_throughput;
-- dut_uw2 (g_user_width => 2) is used for test_border_dilate.
entity tb_canny_window3x3 is
  generic (
    -- Set per VUnit test config in module_canny_window3x3.py: 0 for the
    -- full-throughput test, nonzero (randomized backpressure) otherwise.
    stall_probability_percent : natural;
    runner_cfg : string
  );
end entity tb_canny_window3x3;

architecture tb of tb_canny_window3x3 is

  constant c_clk_period : time := 10 ns;
  constant c_data_width : positive := 8;

  -- dut_uw1: g_user_width => 1.
  constant c_img_width1  : positive := 8;
  constant c_img_height1 : positive := 6;

  -- dut_uw2: g_user_width => 2.
  constant c_img_width2  : positive := 6;
  constant c_img_height2 : positive := 5;

  signal clk   : std_logic := '0';
  signal rst_n : std_logic := '0';

  -- dut_uw1 signals
  signal s_axis1_tvalid, s_axis1_tready, s_axis1_tlast : std_logic;
  signal s_axis1_tdata : std_logic_vector(c_data_width - 1 downto 0);
  signal s_axis1_tuser : std_logic_vector(0 downto 0);

  signal m_axis1_tvalid, m_axis1_tready, m_axis1_tlast : std_logic;
  signal m_axis1_tdata : std_logic_vector(9 * c_data_width - 1 downto 0);
  signal m_axis1_tuser : std_logic_vector(1 downto 0);

  -- dut_uw2 signals
  signal s_axis2_tvalid, s_axis2_tready, s_axis2_tlast : std_logic;
  signal s_axis2_tdata : std_logic_vector(c_data_width - 1 downto 0);
  signal s_axis2_tuser : std_logic_vector(1 downto 0);

  signal m_axis2_tvalid, m_axis2_tready, m_axis2_tlast : std_logic;
  signal m_axis2_tdata : std_logic_vector(9 * c_data_width - 1 downto 0);
  signal m_axis2_tuser : std_logic_vector(1 downto 0);

  constant c_stall_config : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master_uw1 : axi_stream_master_t := new_axi_stream_master(
    data_length  => c_data_width,
    user_length  => 1,
    stall_config => c_stall_config,
    logger       => get_logger("axi_master_uw1")
  );
  constant axi_slave_uw1 : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => 9 * c_data_width,
    user_length  => 2,
    stall_config => c_stall_config,
    logger       => get_logger("axi_slave_uw1")
  );

  constant axi_master_uw2 : axi_stream_master_t := new_axi_stream_master(
    data_length  => c_data_width,
    user_length  => 2,
    stall_config => c_stall_config,
    logger       => get_logger("axi_master_uw2")
  );
  constant axi_slave_uw2 : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => 9 * c_data_width,
    user_length  => 2,
    stall_config => c_stall_config,
    logger       => get_logger("axi_slave_uw2")
  );

  -- True if (row, col) lies within a frame of the given dimensions.
  function in_frame(
    row, col : integer;
    img_width, img_height : positive
  ) return boolean is
  begin
    return row >= 0 and row <= img_height - 1 and col >= 0 and col <= img_width - 1;
  end function;

  function to_sl(cond : boolean) return std_logic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  type frame1_data_t is array(0 to c_img_height1 - 1, 0 to c_img_width1 - 1)
    of std_logic_vector(c_data_width - 1 downto 0);

  type frame2_data_t is array(0 to c_img_height2 - 1, 0 to c_img_width2 - 1)
    of std_logic_vector(c_data_width - 1 downto 0);
  type frame2_border_t is array(0 to c_img_height2 - 1, 0 to c_img_width2 - 1) of std_logic;

begin

  test_runner_watchdog(runner, 2 ms);
  clk <= not clk after c_clk_period / 2;

  rst_n_gen : process
  begin
    rst_n <= '0';
    wait for 3 * c_clk_period;
    rst_n <= '1';
    wait;
  end process;


  ------------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;

    -- Pushes one full raster-scan frame of random data to dut_uw1
    -- (g_user_width => 1, so no incoming border bit exists at all), and
    -- checks every output window/tuser/tlast against an independently
    -- re-derived model of the requirement's masking/dilate/tlast formulas
    -- (not copy-pasted from the RTL under test).
    procedure run_random_data_test(check_throughput : boolean := false) is
      variable frame : frame1_data_t;
      variable in_user : std_logic_vector(0 downto 0);
      variable sof, is_last : std_logic;
      variable expected_data : std_logic_vector(9 * c_data_width - 1 downto 0);
      variable expected_user : std_logic_vector(1 downto 0);
      variable start_time : time;
      variable tap_row, tap_col : integer;
      variable byte_idx : natural;
    begin
      -- Fill the model frame with random data.
      for row in 0 to c_img_height1 - 1 loop
        for col in 0 to c_img_width1 - 1 loop
          frame(row, col) := std_logic_vector(
            to_unsigned(rnd.RandInt(0, 2 ** c_data_width - 1), c_data_width)
          );
        end loop;
      end loop;

      start_time := now;

      for row in 0 to c_img_height1 - 1 loop
        for col in 0 to c_img_width1 - 1 loop
          sof := to_sl(row = 0 and col = 0);
          is_last := to_sl(col = c_img_width1 - 1);
          in_user(0) := sof;

          push_axi_stream(
            net        => net,
            axi_stream => axi_master_uw1,
            tdata      => frame(row, col),
            tlast      => is_last,
            tuser      => in_user
          );

          -- Build the expected window for output position (row, col): every
          -- accepted input beat produces exactly one output beat 1:1, once
          -- primed (checked separately by the full-throughput test's own
          -- timing check; ordering-only here since push/check are queued).
          byte_idx := 0;
          for row_off in -1 to 1 loop
            for col_off in -1 to 1 loop
              tap_row := row + row_off;
              tap_col := col + col_off;

              if in_frame(tap_row, tap_col, c_img_width1, c_img_height1) then
                expected_data(
                  (9 - byte_idx) * c_data_width - 1 downto (8 - byte_idx) * c_data_width
                ) := frame(tap_row, tap_col);
              else
                expected_data(
                  (9 - byte_idx) * c_data_width - 1 downto (8 - byte_idx) * c_data_width
                ) := (others => '0');
              end if;

              byte_idx := byte_idx + 1;
            end loop;
          end loop;

          -- g_user_width => 1: no incoming border bit exists, so the
          -- output border bit is exactly this module's own edge-of-frame
          -- test (no dilate contribution possible).
          expected_user := to_sl(
            row = 0 or row = c_img_height1 - 1 or col = 0 or col = c_img_width1 - 1
          ) & sof;

          check_axi_stream(
            net        => net,
            axi_stream => axi_slave_uw1,
            expected   => expected_data,
            tlast      => is_last,
            tuser      => expected_user,
            blocking   => false,
            msg        => "row=" & to_string(row) & " col=" & to_string(col)
          );
        end loop;
      end loop;

      -- The push/check calls above are non-blocking (they only enqueue);
      -- drain both VCs so this procedure does not return (and so the
      -- throughput measurement below does not sample 'now') until every
      -- enqueued beat has actually been driven/checked on the real bus.
      wait_until_idle(net, as_sync(axi_master_uw1));
      wait_until_idle(net, as_sync(axi_slave_uw1));

      if check_throughput then
        -- Zero stall on all sides: must sustain one output beat per
        -- accepted input beat once primed. Total wall-clock time budget is
        -- the full frame plus the fixed 2*g_img_width+2 fill latency, plus
        -- a small margin for reset/startup.
        check_relation(
          (now - start_time)
            < (c_img_width1 * c_img_height1 + 2 * c_img_width1 + 2 + 10) * c_clk_period,
          "canny_window3x3 did not sustain full throughput at zero stall"
        );
      end if;
    end procedure;

    -- Directed border-dilate test on dut_uw2 (g_user_width => 2): a single
    -- interior incoming border bit is injected, and the expected output
    -- border bit is independently re-derived as this module's own
    -- edge-of-frame test OR'd with a 9-tap OR-reduction of the incoming
    -- border bits (masked to '0' for any out-of-frame tap), so the
    -- OR-dilate is exercised separately from the self-edge test at every
    -- other position in the frame.
    procedure run_border_dilate_test is
      variable frame : frame2_data_t;
      variable border : frame2_border_t;
      variable in_user : std_logic_vector(1 downto 0);
      variable sof, is_last : std_logic;
      variable expected_data : std_logic_vector(9 * c_data_width - 1 downto 0);
      variable expected_user : std_logic_vector(1 downto 0);
      variable tap_row, tap_col : integer;
      variable byte_idx : natural;
      variable border_or, self_edge : std_logic;

      -- A single interior injected border bit, strictly away from every
      -- frame edge, so its dilate footprint (Chebyshev distance <= 1) does
      -- not overlap the self-edge ring at all.
      constant c_border_row : natural := 2;
      constant c_border_col : natural := 3;
    begin
      for row in 0 to c_img_height2 - 1 loop
        for col in 0 to c_img_width2 - 1 loop
          frame(row, col) := std_logic_vector(
            to_unsigned(rnd.RandInt(0, 2 ** c_data_width - 1), c_data_width)
          );
          border(row, col) := to_sl(row = c_border_row and col = c_border_col);
        end loop;
      end loop;

      for row in 0 to c_img_height2 - 1 loop
        for col in 0 to c_img_width2 - 1 loop
          sof := to_sl(row = 0 and col = 0);
          is_last := to_sl(col = c_img_width2 - 1);
          in_user := border(row, col) & sof;

          push_axi_stream(
            net        => net,
            axi_stream => axi_master_uw2,
            tdata      => frame(row, col),
            tlast      => is_last,
            tuser      => in_user
          );

          byte_idx := 0;
          border_or := '0';
          for row_off in -1 to 1 loop
            for col_off in -1 to 1 loop
              tap_row := row + row_off;
              tap_col := col + col_off;

              if in_frame(tap_row, tap_col, c_img_width2, c_img_height2) then
                expected_data(
                  (9 - byte_idx) * c_data_width - 1 downto (8 - byte_idx) * c_data_width
                ) := frame(tap_row, tap_col);
                border_or := border_or or border(tap_row, tap_col);
              else
                expected_data(
                  (9 - byte_idx) * c_data_width - 1 downto (8 - byte_idx) * c_data_width
                ) := (others => '0');
              end if;

              byte_idx := byte_idx + 1;
            end loop;
          end loop;

          self_edge := to_sl(
            row = 0 or row = c_img_height2 - 1 or col = 0 or col = c_img_width2 - 1
          );
          expected_user := (self_edge or border_or) & sof;

          check_axi_stream(
            net        => net,
            axi_stream => axi_slave_uw2,
            expected   => expected_data,
            tlast      => is_last,
            tuser      => expected_user,
            blocking   => false,
            msg        => "row=" & to_string(row) & " col=" & to_string(col)
          );
        end loop;
      end loop;

      -- See the matching comment in run_random_data_test -- drain before
      -- returning so this test actually exercises the real bus.
      wait_until_idle(net, as_sync(axi_master_uw2));
      wait_until_idle(net, as_sync(axi_slave_uw2));
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    wait until rst_n = '1' and rising_edge(clk);

    if run("test_random_data") then
      run_random_data_test;

    elsif run("test_full_throughput") then
      run_random_data_test(check_throughput => true);

    elsif run("test_border_dilate") then
      run_border_dilate_test;

    end if;

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
  axi_stream_master_uw1_inst : entity vunit_lib.axi_stream_master
    generic map (
      master => axi_master_uw1
    )
    port map (
      aclk   => clk,
      tvalid => s_axis1_tvalid,
      tready => s_axis1_tready,
      tdata  => s_axis1_tdata,
      tlast  => s_axis1_tlast,
      tuser  => s_axis1_tuser
    );

  axi_stream_slave_uw1_inst : entity vunit_lib.axi_stream_slave
    generic map (
      slave => axi_slave_uw1
    )
    port map (
      aclk   => clk,
      tvalid => m_axis1_tvalid,
      tready => m_axis1_tready,
      tdata  => m_axis1_tdata,
      tlast  => m_axis1_tlast,
      tuser  => m_axis1_tuser
    );

  dut_uw1 : entity work.canny_window3x3
    generic map (
      g_img_width  => c_img_width1,
      g_img_height => c_img_height1,
      g_data_width => c_data_width,
      g_user_width => 1
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => s_axis1_tvalid,
      s_axis_tready => s_axis1_tready,
      s_axis_tdata  => s_axis1_tdata,
      s_axis_tuser  => s_axis1_tuser,
      s_axis_tlast  => s_axis1_tlast,

      m_axis_tvalid => m_axis1_tvalid,
      m_axis_tready => m_axis1_tready,
      m_axis_tdata  => m_axis1_tdata,
      m_axis_tuser  => m_axis1_tuser,
      m_axis_tlast  => m_axis1_tlast
    );


  ------------------------------------------------------------------------------
  axi_stream_master_uw2_inst : entity vunit_lib.axi_stream_master
    generic map (
      master => axi_master_uw2
    )
    port map (
      aclk   => clk,
      tvalid => s_axis2_tvalid,
      tready => s_axis2_tready,
      tdata  => s_axis2_tdata,
      tlast  => s_axis2_tlast,
      tuser  => s_axis2_tuser
    );

  axi_stream_slave_uw2_inst : entity vunit_lib.axi_stream_slave
    generic map (
      slave => axi_slave_uw2
    )
    port map (
      aclk   => clk,
      tvalid => m_axis2_tvalid,
      tready => m_axis2_tready,
      tdata  => m_axis2_tdata,
      tlast  => m_axis2_tlast,
      tuser  => m_axis2_tuser
    );

  dut_uw2 : entity work.canny_window3x3
    generic map (
      g_img_width  => c_img_width2,
      g_img_height => c_img_height2,
      g_data_width => c_data_width,
      g_user_width => 2
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

      s_axis_tvalid => s_axis2_tvalid,
      s_axis_tready => s_axis2_tready,
      s_axis_tdata  => s_axis2_tdata,
      s_axis_tuser  => s_axis2_tuser,
      s_axis_tlast  => s_axis2_tlast,

      m_axis_tvalid => m_axis2_tvalid,
      m_axis_tready => m_axis2_tready,
      m_axis_tdata  => m_axis2_tdata,
      m_axis_tuser  => m_axis2_tuser,
      m_axis_tlast  => m_axis2_tlast
    );

end architecture tb;
