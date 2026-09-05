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

-- TDD testbench for modules/canny/src/canny_nms.vhd, written before the
-- --@ markers in that file are resolved (per the project's TDD policy in
-- shared/Vunit.md #15). Uses VUnit's raw axi_stream_master/axi_stream_slave
-- verification components directly (not the hdl-modules bfm.* wrappers),
-- because bfm.axi_stream_master/slave require user_width mod 8 = 0 and this
-- module's tuser is 2 bits -- see shared/Vunit.md #12.
entity tb_canny_nms is
  generic (
    -- Set per VUnit test config in module_canny.py: 0 for the
    -- full-throughput test, nonzero (randomized backpressure) otherwise.
    stall_probability_percent : natural;
    runner_cfg : string
  );
end entity tb_canny_nms;

architecture tb of tb_canny_nms is

  constant clk_period : time := 10 ns;
  constant mag_width  : positive := 11;
  constant data_width : positive := 9 * mag_width + 2;

  signal clk   : std_logic := '0';
  signal reset : std_logic := '1';

  signal s_axis_tvalid, s_axis_tready, s_axis_tlast : std_logic;
  signal s_axis_tdata : std_logic_vector(data_width - 1 downto 0);
  signal s_axis_tuser : std_logic_vector(1 downto 0);

  signal m_axis_tvalid, m_axis_tready, m_axis_tlast : std_logic;
  signal m_axis_tdata : std_logic_vector(mag_width - 1 downto 0);
  signal m_axis_tuser : std_logic_vector(1 downto 0);

  constant stall_config : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master : axi_stream_master_t := new_axi_stream_master(
    data_length  => data_width,
    user_length  => 2,
    stall_config => stall_config,
    logger       => get_logger("axi_master")
  );
  constant axi_slave : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => mag_width,
    user_length  => 2,
    stall_config => stall_config,
    logger       => get_logger("axi_slave")
  );

  -- Tap order matches canny_window3x3's row-major packing: 0=tl, 1=tm,
  -- 2=tr, 3=ml, 4=mm, 5=mr, 6=bl, 7=bm, 8=br.
  type taps9_t is array(0 to 8) of std_logic_vector(mag_width - 1 downto 0);

  function to_sl(cond : boolean) return std_logic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  -- Packs the 9x11-bit magnitude window and the 2-bit direction sector
  -- into a single s_axis_tdata vector (101 bits), per
  -- doc/canny_nms_req.md's structural section.
  function pack_tdata(taps : taps9_t; direction : std_logic_vector(1 downto 0))
    return std_logic_vector is
  begin
    return taps(0) & taps(1) & taps(2) & taps(3) & taps(4) & taps(5)
      & taps(6) & taps(7) & taps(8) & direction;
  end function;

  -- Independently re-derived expected-output model from
  -- doc/canny_nms_req.md's exact functional description (not copy-pasted
  -- from the RTL under test): non-strict ">=" local-maximum test along
  -- the direction-sector-selected neighbor pair, forced to 0 at the
  -- border.
  function compute_expected(
    taps      : taps9_t;
    direction : std_logic_vector(1 downto 0);
    border    : std_logic
  ) return std_logic_vector is
    variable mm, neighbor_a, neighbor_b : unsigned(mag_width - 1 downto 0);
  begin
    mm := unsigned(taps(4));

    case direction is
      when "00" =>
        neighbor_a := unsigned(taps(3)); -- w_ml
        neighbor_b := unsigned(taps(5)); -- w_mr
      when "01" =>
        neighbor_a := unsigned(taps(2)); -- w_tr
        neighbor_b := unsigned(taps(6)); -- w_bl
      when "10" =>
        neighbor_a := unsigned(taps(1)); -- w_tm
        neighbor_b := unsigned(taps(7)); -- w_bm
      when others =>
        neighbor_a := unsigned(taps(0)); -- w_tl
        neighbor_b := unsigned(taps(8)); -- w_br
    end case;

    if border = '1' then
      return (mag_width - 1 downto 0 => '0');
    elsif mm >= neighbor_a and mm >= neighbor_b then
      return std_logic_vector(mm);
    else
      return (mag_width - 1 downto 0 => '0');
    end if;
  end function;

begin

  test_runner_watchdog(runner, 2 ms);
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
    variable rnd : RandomPType;

    -- Pushes one beat and queues its expected check, given explicit taps/
    -- direction/border/sof/is_last. Shared by every test case below so the
    -- packing/expected-model logic is only written once.
    procedure push_and_check(
      taps      : taps9_t;
      direction : std_logic_vector(1 downto 0);
      border    : std_logic;
      sof       : std_logic;
      is_last   : std_logic;
      msg       : string := ""
    ) is
      variable in_user, expected_user : std_logic_vector(1 downto 0);
      variable expected_data : std_logic_vector(mag_width - 1 downto 0);
    begin
      in_user := border & sof;

      push_axi_stream(
        net        => net,
        axi_stream => axi_master,
        tdata      => pack_tdata(taps, direction),
        tlast      => is_last,
        tuser      => in_user
      );

      expected_data := compute_expected(taps, direction, border);
      expected_user := in_user;

      check_axi_stream(
        net        => net,
        axi_stream => axi_slave,
        expected   => expected_data,
        tlast      => is_last,
        tuser      => expected_user,
        blocking   => false,
        msg        => msg
      );
    end procedure;

    -- Directed coverage of all 4 direction sectors: local-maximum-kept,
    -- local-non-maximum-suppressed, and tie-kept (non-strict ">=") cases.
    procedure run_direction_sectors_test is
      type sector_t is array(natural range <>) of std_logic_vector(1 downto 0);
      constant sectors : sector_t(0 to 3) := ("00", "01", "10", "11");
      variable taps : taps9_t;
      variable beat_idx : natural;
    begin
      beat_idx := 0;
      for i in sectors'range loop
        -- Fill every tap with a distinct low value so only the two
        -- direction-relevant neighbors and the center matter.
        for t in taps'range loop
          taps(t) := std_logic_vector(to_unsigned(10 + t, mag_width));
        end loop;

        -- Case 1: local maximum -- center strictly greater than both
        -- direction-relevant neighbors -- must be kept.
        taps(4) := std_logic_vector(to_unsigned(200, mag_width));
        push_and_check(
          taps, sectors(i), '0', to_sl(beat_idx = 0), '0',
          "sector=" & to_string(sectors(i)) & " local_max"
        );
        beat_idx := beat_idx + 1;

        -- Case 2: local non-maximum -- one direction-relevant neighbor is
        -- strictly greater than the center -- must be suppressed.
        for t in taps'range loop
          taps(t) := std_logic_vector(to_unsigned(10 + t, mag_width));
        end loop;
        taps(4) := std_logic_vector(to_unsigned(5, mag_width));
        push_and_check(
          taps, sectors(i), '0', '0', '0',
          "sector=" & to_string(sectors(i)) & " suppressed"
        );
        beat_idx := beat_idx + 1;

        -- Case 3: tie -- center exactly equal to both direction-relevant
        -- neighbors -- must be kept (non-strict ">=" per requirement).
        for t in taps'range loop
          taps(t) := std_logic_vector(to_unsigned(50, mag_width));
        end loop;
        push_and_check(
          taps, sectors(i), '0', '0', to_sl(i = sectors'high),
          "sector=" & to_string(sectors(i)) & " tie"
        );
        beat_idx := beat_idx + 1;
      end loop;
    end procedure;

    -- Random 9-tap windows and random direction per beat; expected result
    -- computed via compute_expected (independently re-derived model).
    procedure run_random_data_test(num_beats : positive; check_throughput : boolean := false) is
      variable taps : taps9_t;
      variable direction : std_logic_vector(1 downto 0);
      variable start_time : time;
    begin
      start_time := now;

      for beat in 0 to num_beats - 1 loop
        for t in taps'range loop
          taps(t) := std_logic_vector(
            to_unsigned(rnd.RandInt(0, 2 ** mag_width - 1), mag_width)
          );
        end loop;
        direction := std_logic_vector(to_unsigned(rnd.RandInt(0, 3), 2));

        push_and_check(
          taps, direction, '0',
          to_sl(beat = 0), to_sl(beat = num_beats - 1),
          "random beat=" & to_string(beat)
        );
      end loop;

      -- push_axi_stream/check_axi_stream (non-blocking) only enqueue
      -- commands on the actors' mailboxes; they return long before the
      -- corresponding bus activity has actually happened. Drain both VCs
      -- before measuring elapsed time or letting the caller proceed to
      -- test_runner_cleanup, otherwise the checks queued here would never
      -- actually run against real bus activity.
      wait_until_idle(net, as_sync(axi_master));
      wait_until_idle(net, as_sync(axi_slave));

      if check_throughput then
        -- Zero stall on both sides, fixed 1-cycle handshake_pipeline
        -- latency: must sustain one output beat per accepted input beat.
        check_relation(
          (now - start_time) < (num_beats + 10) * clk_period,
          "canny_nms did not sustain full throughput at zero stall"
        );
      end if;
    end procedure;

    -- Border-forces-zero: some beats have the incoming border bit set;
    -- expected output must be 0 regardless of the compare result, while
    -- tuser(1) itself still reads back '1' (passthrough).
    procedure run_border_test is
      variable taps : taps9_t;
    begin
      for beat in 0 to 7 loop
        for t in taps'range loop
          taps(t) := std_logic_vector(
            to_unsigned(rnd.RandInt(0, 2 ** mag_width - 1), mag_width)
          );
        end loop;
        -- Force the center to be an unambiguous local maximum along every
        -- sector, so a wrongly-not-suppressed result at the border can
        -- only be explained by the border forcing itself being broken,
        -- not an unrelated compare-formula edge case.
        taps(4) := std_logic_vector(to_unsigned(2 ** mag_width - 1, mag_width));

        push_and_check(
          taps, "00", '1', to_sl(beat = 0), to_sl(beat = 7),
          "border beat=" & to_string(beat)
        );
      end loop;

      -- One trailing non-border beat to confirm normal passthrough of
      -- tuser(1) = '0' still works right after a run of border beats.
      for t in taps'range loop
        taps(t) := std_logic_vector(to_unsigned(1, mag_width));
      end loop;
      taps(4) := std_logic_vector(to_unsigned(2 ** mag_width - 1, mag_width));
      push_and_check(taps, "00", '0', '0', '1', "border trailing non-border beat");
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    wait until reset = '0' and rising_edge(clk);

    if run("test_direction_sectors") then
      run_direction_sectors_test;

    elsif run("test_random_data") then
      run_random_data_test(num_beats => 200);

    elsif run("test_border_forces_zero") then
      run_border_test;

    elsif run("test_full_throughput") then
      run_random_data_test(num_beats => 300, check_throughput => true);

    end if;

    -- Same rationale as inside run_random_data_test above: drain both VCs
    -- so every queued push/check has actually run against real bus
    -- activity before the simulation is allowed to end.
    wait_until_idle(net, as_sync(axi_master));
    wait_until_idle(net, as_sync(axi_slave));

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
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

  dut : entity work.canny_nms
    port map (
      clk   => clk,
      reset => reset,

      s_axis_tvalid => s_axis_tvalid,
      s_axis_tready => s_axis_tready,
      s_axis_tdata  => s_axis_tdata,
      s_axis_tuser  => s_axis_tuser,
      s_axis_tlast  => s_axis_tlast,

      m_axis_tvalid => m_axis_tvalid,
      m_axis_tready => m_axis_tready,
      m_axis_tdata  => m_axis_tdata,
      m_axis_tuser  => m_axis_tuser,
      m_axis_tlast  => m_axis_tlast
    );

end architecture tb;
