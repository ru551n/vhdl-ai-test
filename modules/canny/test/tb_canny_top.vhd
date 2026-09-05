library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.axi_stream_pkg.all;
use vunit_lib.sync_pkg.all;
use vunit_lib.integer_array_pkg.all;

-- IP-level integration testbench for modules/canny/src/canny_top.vhd,
-- driven by the Python golden model (canny_model.canny_pipeline) via
-- pre_config/post_check per shared/Vunit.md "Python reference models":
--
-- module_canny.py's pre_config() writes a random stimulus frame
-- ("stimulus.csv") and the model's precomputed expected edge frame
-- ("expected.csv") into this config's own output_path before the
-- simulation runs. This testbench only reads stimulus.csv back (via
-- integer_array_pkg.load_csv), drives it into the DUT, captures the DUT's
-- own output into a same-shaped integer_array_t, and dumps it as
-- "result.csv" (via integer_array_pkg.save_csv) -- it never computes an
-- expected pixel/edge value itself. module_canny.py's post_check()
-- then loads expected.csv/result.csv and does the actual value-for-value
-- comparison in Python, with a row/col diagnostic on mismatch.
--
-- Framing (tlast = end-of-line, tuser(0) = start-of-frame) is re-derived
-- here from the loop position and checked directly in VHDL (the golden
-- model only owns "is the computed edge value correct", per
-- shared/Vunit.md's Python-reference-model rule).
--
-- Uses VUnit's raw axi_stream_master/axi_stream_slave directly (not the
-- hdl-modules bfm.* wrappers), per shared/Vunit.md's user_width mod 8 = 0
-- caveat -- this DUT's tuser is 1 bit wide.
entity tb_canny_top is
  generic (
    -- Set per VUnit test config in module_canny.py.
    stall_probability_percent : natural;
    img_width   : positive;
    img_height  : positive;
    thresh_low  : natural;
    thresh_high : natural;
    -- Auto-filled by VUnit with this config's own output directory (see
    -- vunit/test/suites.py: any testbench declaring an "output_path"
    -- generic gets it set to the same path passed to pre_config/post_check).
    output_path : string := "";
    runner_cfg  : string
  );
end entity tb_canny_top;

architecture tb of tb_canny_top is

  constant clk_period : time := 10 ns;
  constant data_width : positive := 8;

  signal clk   : std_logic := '0';
  signal reset : std_logic := '1';

  signal s_axis_tvalid, s_axis_tready, s_axis_tlast : std_logic;
  signal s_axis_tdata : std_logic_vector(data_width - 1 downto 0);
  signal s_axis_tuser : std_logic_vector(0 downto 0);

  signal m_axis_tvalid, m_axis_tready, m_axis_tlast : std_logic;
  signal m_axis_tdata : std_logic_vector(data_width - 1 downto 0);
  signal m_axis_tuser : std_logic_vector(0 downto 0);

  constant stall_config : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master : axi_stream_master_t := new_axi_stream_master(
    data_length  => data_width,
    user_length  => 1,
    stall_config => stall_config,
    logger       => get_logger("axi_master")
  );
  constant axi_slave : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => data_width,
    user_length  => 1,
    stall_config => stall_config,
    logger       => get_logger("axi_slave")
  );

  function to_sl(cond : boolean) return std_logic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

begin

  test_runner_watchdog(runner, 5 ms);
  clk <= not clk after clk_period / 2;

  reset_gen : process
  begin
    reset <= '1';
    wait for 3 * clk_period;
    reset <= '0';
    wait;
  end process;

  ------------------------------------------------------------------------------
  main : process
    variable stim_arr   : integer_array_t;
    variable result_arr : integer_array_t;
    variable pixel      : integer;

    -- pop_axi_stream out params: only tdata/tlast/tuser are meaningful here
    -- (id_length/dest_length default to 0 -> tid_v/tdest_v are null-range).
    variable rx_data  : std_logic_vector(data_width - 1 downto 0);
    variable rx_tlast : std_logic;
    variable rx_tkeep : std_logic_vector(data_width / 8 - 1 downto 0);
    variable rx_tstrb : std_logic_vector(data_width / 8 - 1 downto 0);
    variable rx_tid   : std_logic_vector(0 downto 1);
    variable rx_tdest : std_logic_vector(0 downto 1);
    variable rx_tuser : std_logic_vector(0 downto 0);
  begin
    test_runner_setup(runner, runner_cfg);

    wait until reset = '0' and rising_edge(clk);

    if run("test_full_pipeline") then
      stim_arr := load_csv(output_path & "stimulus.csv", bit_width => 8, is_signed => false);
      assert width(stim_arr) = img_width and height(stim_arr) = img_height
        report "tb_canny_top: stimulus.csv shape does not match img_width/img_height"
        severity failure;

      result_arr := new_2d(
        width => img_width, height => img_height, bit_width => 8, is_signed => false
      );

      -- Drive the whole frame in, non-blocking (queues every beat
      -- up-front; the DUT's own handshaking/backpressure absorbs the
      -- timing) -- per shared/Vunit.md #12/#15.
      for row in 0 to img_height - 1 loop
        for col in 0 to img_width - 1 loop
          pixel := get(stim_arr, col, row);
          push_axi_stream(
            net        => net,
            axi_stream => axi_master,
            tdata      => std_logic_vector(to_unsigned(pixel, data_width)),
            tlast      => to_sl(col = img_width - 1),
            tuser      => (0 => to_sl(row = 0 and col = 0))
          );
        end loop;
      end loop;

      -- Capture the DUT's own output beat by beat (blocking pop -- the
      -- checking process here doubles as the drain, so no separate
      -- wait_until_idle race is possible); check framing against the
      -- position derived here, and stash the data value for post_check's
      -- golden-model comparison (this TB never computes an expected edge
      -- value itself).
      for row in 0 to img_height - 1 loop
        for col in 0 to img_width - 1 loop
          pop_axi_stream(
            net        => net,
            axi_stream => axi_slave,
            tdata      => rx_data,
            tlast      => rx_tlast,
            tkeep      => rx_tkeep,
            tstrb      => rx_tstrb,
            tid        => rx_tid,
            tdest      => rx_tdest,
            tuser      => rx_tuser
          );

          check_equal(
            rx_tlast, to_sl(col = img_width - 1),
            msg => "tlast at row=" & to_string(row) & " col=" & to_string(col)
          );
          check_equal(
            rx_tuser(0), to_sl(row = 0 and col = 0),
            msg => "tuser(0)/SOF at row=" & to_string(row) & " col=" & to_string(col)
          );

          set(result_arr, col, row, to_integer(unsigned(rx_data)));
        end loop;
      end loop;

      save_csv(result_arr, output_path & "result.csv");

      -- Both VCs' queues are already fully drained by the sequential
      -- push/pop loops above (one beat pushed, then all beats popped in
      -- strict order) -- these calls are a cheap extra safety margin, not
      -- load-bearing.
      wait_until_idle(net, as_sync(axi_master));
      wait_until_idle(net, as_sync(axi_slave));
    end if;

    test_runner_cleanup(runner);
    wait;
  end process;

  ------------------------------------------------------------------------------
  dut : entity work.canny_top
    generic map (
      img_width   => img_width,
      img_height  => img_height,
      thresh_low  => thresh_low,
      thresh_high => thresh_high
    )
    port map (
      clk   => clk,
      reset => reset,

      s_axis_tvalid => s_axis_tvalid,
      s_axis_tready => s_axis_tready,
      s_axis_tdata  => s_axis_tdata,
      s_axis_tlast  => s_axis_tlast,
      s_axis_tuser  => s_axis_tuser,

      m_axis_tvalid => m_axis_tvalid,
      m_axis_tready => m_axis_tready,
      m_axis_tdata  => m_axis_tdata,
      m_axis_tlast  => m_axis_tlast,
      m_axis_tuser  => m_axis_tuser
    );

  axi_stream_master_inst : entity vunit_lib.axi_stream_master
    generic map (
      master => axi_master
    )
    port map (
      aclk   => clk,
      tvalid => s_axis_tvalid,
      tready => s_axis_tready,
      tdata  => s_axis_tdata,
      tlast  => s_axis_tlast,
      tuser  => s_axis_tuser
    );

  axi_stream_slave_inst : entity vunit_lib.axi_stream_slave
    generic map (
      slave => axi_slave
    )
    port map (
      aclk   => clk,
      tvalid => m_axis_tvalid,
      tready => m_axis_tready,
      tdata  => m_axis_tdata,
      tlast  => m_axis_tlast,
      tuser  => m_axis_tuser
    );

end architecture tb;
