library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
context vunit_lib.vc_context;
use vunit_lib.axi_stream_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

-- TDD testbench for modules/canny/src/canny_sobel3x3.vhd, written
-- before the --@ markers in that file are resolved (per the project's TDD
-- policy in shared/Vunit.md #15). Uses VUnit's raw axi_stream_master/
-- axi_stream_slave verification components directly (not the hdl-modules
-- bfm.* wrappers), because bfm.axi_stream_master/slave require
-- user_width mod 8 = 0 and this module's s_axis_tuser/m_axis_*_tuser are
-- 2 bits wide -- see shared/Vunit.md #12.
--
-- The two output forks (mag, dir) get independently randomized
-- stall_config (stall_probability_percent_mag / _dir), per the module's
-- whole reason for existing: handshake_splitter's per-output sticky
-- bookkeeping must behave correctly when the two forks' consumers stall
-- for different durations, not just when both are stalled identically or
-- not at all.
entity tb_canny_sobel3x3 is
  generic (
    -- Set per VUnit test config in module_canny.py: all 0 for the
    -- full-throughput test, independently randomized nonzero otherwise.
    stall_probability_percent_in  : natural;
    stall_probability_percent_mag : natural;
    stall_probability_percent_dir : natural;
    runner_cfg : string
  );
end entity tb_canny_sobel3x3;

architecture tb of tb_canny_sobel3x3 is

  constant clk_period : time := 10 ns;

  constant tap_width  : positive := 8;
  constant window_width : positive := 9 * tap_width;
  constant mag_width  : positive := 11;
  constant dir_width  : positive := 2;
  constant user_width : positive := 2;

  signal clk   : std_logic := '0';
  signal reset : std_logic := '1';

  signal s_axis_tvalid, s_axis_tready, s_axis_tlast : std_logic;
  signal s_axis_tdata : std_logic_vector(window_width - 1 downto 0);
  signal s_axis_tuser : std_logic_vector(user_width - 1 downto 0);

  signal m_axis_mag_tvalid, m_axis_mag_tready, m_axis_mag_tlast : std_logic;
  signal m_axis_mag_tdata : std_logic_vector(mag_width - 1 downto 0);
  signal m_axis_mag_tuser : std_logic_vector(user_width - 1 downto 0);

  signal m_axis_dir_tvalid, m_axis_dir_tready, m_axis_dir_tlast : std_logic;
  signal m_axis_dir_tdata : std_logic_vector(dir_width - 1 downto 0);
  signal m_axis_dir_tuser : std_logic_vector(user_width - 1 downto 0);

  constant stall_config_in : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent_in) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );
  constant stall_config_mag : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent_mag) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );
  constant stall_config_dir : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent_dir) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master_in : axi_stream_master_t := new_axi_stream_master(
    data_length  => window_width,
    user_length  => user_width,
    stall_config => stall_config_in,
    logger       => get_logger("axi_master_in")
  );
  constant axi_slave_mag : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => mag_width,
    user_length  => user_width,
    stall_config => stall_config_mag,
    logger       => get_logger("axi_slave_mag")
  );
  constant axi_slave_dir : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => dir_width,
    user_length  => user_width,
    stall_config => stall_config_dir,
    logger       => get_logger("axi_slave_dir")
  );

  function to_slv8(v : natural) return std_logic_vector is
  begin
    return std_logic_vector(to_unsigned(v, tap_width));
  end function;

  function to_sl(cond : boolean) return std_logic is
  begin
    if cond then
      return '1';
    end if;
    return '0';
  end function;

  -- Same packing as canny_gaussian3x3's input: w_tl (MSB) .. w_br (LSB).
  function pack_window(
    tl, tm, tr, ml, mm, mr, bl, bm, br : natural
  ) return std_logic_vector is
  begin
    return to_slv8(tl) & to_slv8(tm) & to_slv8(tr) & to_slv8(ml) & to_slv8(mm)
      & to_slv8(mr) & to_slv8(bl) & to_slv8(bm) & to_slv8(br);
  end function;

  -- Independently re-derived from doc/canny_sobel3x3_req.md's Functional
  -- Description (not copy-pasted from the RTL under test).
  function calc_gx(tl, tm, tr, ml, mm, mr, bl, bm, br : natural) return integer is
  begin
    return (tr + 2 * mr + br) - (tl + 2 * ml + bl);
  end function;

  function calc_gy(tl, tm, tr, ml, mm, mr, bl, bm, br : natural) return integer is
  begin
    return (bl + 2 * bm + br) - (tl + 2 * tm + tr);
  end function;

  -- The "00" branch is checked first, so ax=ay=0 resolves to "00" per the
  -- requirement's explicit tie-break rule.
  function calc_dir(gx, gy : integer) return std_logic_vector is
    variable ax, ay : natural;
  begin
    ax := abs(gx);
    ay := abs(gy);

    if ay <= ax / 2 then
      return "00";
    elsif ax <= ay / 2 then
      return "10";
    elsif (gx >= 0) = (gy >= 0) then
      return "01";
    else
      return "11";
    end if;
  end function;

  function calc_mag(gx, gy : integer) return std_logic_vector is
  begin
    return std_logic_vector(to_unsigned(abs(gx) + abs(gy), mag_width));
  end function;

begin

  test_runner_watchdog(runner, 10 ms);
  clk <= not clk after clk_period / 2;

  reset_gen : process
  begin
    reset <= '1';
    wait for 3 * clk_period;
    reset <= '0';
    wait;
  end process;

  -- Requirement: both fork outputs are driven from the same register, so
  -- their tuser/tlast can never legally disagree -- checked on every cycle
  -- where both happen to be valid (independent of whether they are
  -- simultaneously accepted, since the payload must stay stable regardless).
  fork_synchronization_check : process
  begin
    wait until rising_edge(clk);

    if reset = '0' and m_axis_mag_tvalid = '1' and m_axis_dir_tvalid = '1' then
      assert m_axis_mag_tuser = m_axis_dir_tuser and m_axis_mag_tlast = m_axis_dir_tlast
        report "canny_sobel3x3: fork desynchronization -- m_axis_mag and " &
          "m_axis_dir tuser/tlast disagree while both valid"
        severity error;
    end if;
  end process;


  ------------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;

    -- Pushes num_beats random windows, each with an independently
    -- randomized border bit, and checks the corresponding mag/dir beats
    -- (independent stall_config per fork) against calc_mag/calc_dir/border
    -- forcing.
    procedure run_random_data_test(
      num_beats : positive;
      check_throughput : boolean := false
    ) is
      variable tl, tm, tr, ml, mm, mr, bl, bm, br : natural;
      variable border, sof, is_last : std_logic;
      variable gx, gy : integer;
      variable expected_mag : std_logic_vector(mag_width - 1 downto 0);
      variable expected_dir : std_logic_vector(dir_width - 1 downto 0);
      variable expected_user : std_logic_vector(user_width - 1 downto 0);
      variable start_time : time;
    begin
      start_time := now;

      for beat in 0 to num_beats - 1 loop
        tl := rnd.RandInt(0, 255);
        tm := rnd.RandInt(0, 255);
        tr := rnd.RandInt(0, 255);
        ml := rnd.RandInt(0, 255);
        mm := rnd.RandInt(0, 255);
        mr := rnd.RandInt(0, 255);
        bl := rnd.RandInt(0, 255);
        bm := rnd.RandInt(0, 255);
        br := rnd.RandInt(0, 255);

        border := to_sl(rnd.RandInt(0, 99) < 15);
        sof := to_sl(beat = 0);
        is_last := to_sl(beat mod 8 = 7);

        push_axi_stream(
          net        => net,
          axi_stream => axi_master_in,
          tdata      => pack_window(tl, tm, tr, ml, mm, mr, bl, bm, br),
          tlast      => is_last,
          tuser      => border & sof
        );

        gx := calc_gx(tl, tm, tr, ml, mm, mr, bl, bm, br);
        gy := calc_gy(tl, tm, tr, ml, mm, mr, bl, bm, br);

        if border = '1' then
          expected_mag := (others => '0');
          expected_dir := "00";
        else
          expected_mag := calc_mag(gx, gy);
          expected_dir := calc_dir(gx, gy);
        end if;

        expected_user := border & sof;

        check_axi_stream(
          net        => net,
          axi_stream => axi_slave_mag,
          expected   => expected_mag,
          tlast      => is_last,
          tuser      => expected_user,
          blocking   => false,
          msg        => "mag beat=" & to_string(beat)
        );

        check_axi_stream(
          net        => net,
          axi_stream => axi_slave_dir,
          expected   => expected_dir,
          tlast      => is_last,
          tuser      => expected_user,
          blocking   => false,
          msg        => "dir beat=" & to_string(beat)
        );
      end loop;

      -- push_axi_stream/check_axi_stream(blocking => false) only enqueue
      -- messages for the VC processes; without this the main process would
      -- reach test_runner_cleanup before any of the queued transactions
      -- (and their checks) have actually happened on the bus.
      wait_until_idle(net, as_sync(axi_master_in));
      wait_until_idle(net, as_sync(axi_slave_mag));
      wait_until_idle(net, as_sync(axi_slave_dir));

      if check_throughput then
        -- Zero stall on all three sides: must sustain one output beat per
        -- accepted input beat once primed (1 cycle of registration latency).
        check_relation(
          (now - start_time) < (num_beats + 10) * clk_period,
          "canny_sobel3x3 did not sustain full throughput at zero stall"
        );
      end if;
    end procedure;

    -- Directed case: one beat per direction sector (see
    -- doc/canny_sobel3x3_proposal.md "Algorithms" for the hand-derived tap
    -- values), plus the ax=ay=0 tie-break, plus a border-forces-zero beat.
    procedure run_direction_sectors_test is
      type tap_row_t is array(0 to 8) of natural;
      -- Order: tl, tm, tr, ml, mm, mr, bl, bm, br.
      type case_t is record
        taps        : tap_row_t;
        expected_dir : std_logic_vector(1 downto 0);
        border      : std_logic;
      end record;
      type case_arr_t is array(natural range <>) of case_t;

      constant cases : case_arr_t(0 to 5) := (
        -- Pure horizontal gradient (right column high) -> "00".
        0 => (taps => (0, 0, 255, 0, 0, 255, 0, 0, 255), expected_dir => "00", border => '0'),
        -- Pure vertical gradient (bottom row high) -> "10".
        1 => (taps => (0, 0, 0, 0, 0, 0, 255, 255, 255), expected_dir => "10", border => '0'),
        -- Gx=+300 (mr), Gy=+300 (bm), same sign -> "01".
        2 => (taps => (0, 0, 0, 0, 0, 150, 0, 150, 0), expected_dir => "01", border => '0'),
        -- Gx=+300 (mr), Gy=-300 (tm), opposite sign -> "11".
        3 => (taps => (0, 150, 0, 0, 0, 150, 0, 0, 0), expected_dir => "11", border => '0'),
        -- All taps equal (Gx=Gy=0, ax=ay=0) -> "00" tie-break.
        4 => (taps => (77, 77, 77, 77, 77, 77, 77, 77, 77), expected_dir => "00", border => '0'),
        -- Same as case 1 (would be "10") but with border forced -> "00"/mag=0.
        5 => (taps => (0, 0, 0, 0, 0, 0, 255, 255, 255), expected_dir => "00", border => '1')
      );

      variable gx, gy : integer;
      variable expected_mag : std_logic_vector(mag_width - 1 downto 0);
      variable expected_user : std_logic_vector(user_width - 1 downto 0);
      variable sof, is_last : std_logic;
    begin
      for i in cases'range loop
        sof := to_sl(i = 0);
        is_last := to_sl(i = cases'high);

        push_axi_stream(
          net        => net,
          axi_stream => axi_master_in,
          tdata      => pack_window(
            cases(i).taps(0), cases(i).taps(1), cases(i).taps(2),
            cases(i).taps(3), cases(i).taps(4), cases(i).taps(5),
            cases(i).taps(6), cases(i).taps(7), cases(i).taps(8)
          ),
          tlast => is_last,
          tuser => cases(i).border & sof
        );

        gx := calc_gx(
          cases(i).taps(0), cases(i).taps(1), cases(i).taps(2),
          cases(i).taps(3), cases(i).taps(4), cases(i).taps(5),
          cases(i).taps(6), cases(i).taps(7), cases(i).taps(8)
        );
        gy := calc_gy(
          cases(i).taps(0), cases(i).taps(1), cases(i).taps(2),
          cases(i).taps(3), cases(i).taps(4), cases(i).taps(5),
          cases(i).taps(6), cases(i).taps(7), cases(i).taps(8)
        );

        -- Self-check: the hand-derived expected_dir in cases must agree
        -- with the independently-formulated calc_dir reference.
        assert (cases(i).border = '1') or (calc_dir(gx, gy) = cases(i).expected_dir)
          report "test bug: hand-derived direction for case " & to_string(i) &
            " disagrees with calc_dir reference"
          severity failure;

        expected_user := cases(i).border & sof;

        if cases(i).border = '1' then
          expected_mag := (others => '0');
        else
          expected_mag := calc_mag(gx, gy);
        end if;

        check_axi_stream(
          net        => net,
          axi_stream => axi_slave_mag,
          expected   => expected_mag,
          tlast      => is_last,
          tuser      => expected_user,
          blocking   => false,
          msg        => "sector mag case=" & to_string(i)
        );

        check_axi_stream(
          net        => net,
          axi_stream => axi_slave_dir,
          expected   => cases(i).expected_dir,
          tlast      => is_last,
          tuser      => expected_user,
          blocking   => false,
          msg        => "sector dir case=" & to_string(i)
        );
      end loop;

      wait_until_idle(net, as_sync(axi_master_in));
      wait_until_idle(net, as_sync(axi_slave_mag));
      wait_until_idle(net, as_sync(axi_slave_dir));
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    wait until reset = '0' and rising_edge(clk);

    if run("test_random_data") then
      run_random_data_test(num_beats => 300);

    elsif run("test_full_throughput") then
      run_random_data_test(num_beats => 500, check_throughput => true);

    elsif run("test_direction_sectors") then
      run_direction_sectors_test;

    elsif run("test_border") then
      run_random_data_test(num_beats => 100);

    end if;

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
  axi_stream_master_in_inst : entity vunit_lib.axi_stream_master
    generic map (
      master => axi_master_in
    )
    port map (
      aclk   => clk,
      tvalid => s_axis_tvalid,
      tready => s_axis_tready,
      tdata  => s_axis_tdata,
      tlast  => s_axis_tlast,
      tuser  => s_axis_tuser
    );

  axi_stream_slave_mag_inst : entity vunit_lib.axi_stream_slave
    generic map (
      slave => axi_slave_mag
    )
    port map (
      aclk   => clk,
      tvalid => m_axis_mag_tvalid,
      tready => m_axis_mag_tready,
      tdata  => m_axis_mag_tdata,
      tlast  => m_axis_mag_tlast,
      tuser  => m_axis_mag_tuser
    );

  axi_stream_slave_dir_inst : entity vunit_lib.axi_stream_slave
    generic map (
      slave => axi_slave_dir
    )
    port map (
      aclk   => clk,
      tvalid => m_axis_dir_tvalid,
      tready => m_axis_dir_tready,
      tdata  => m_axis_dir_tdata,
      tlast  => m_axis_dir_tlast,
      tuser  => m_axis_dir_tuser
    );

  dut : entity work.canny_sobel3x3
    port map (
      clk   => clk,
      reset => reset,

      s_axis_tvalid => s_axis_tvalid,
      s_axis_tready => s_axis_tready,
      s_axis_tdata  => s_axis_tdata,
      s_axis_tuser  => s_axis_tuser,
      s_axis_tlast  => s_axis_tlast,

      m_axis_mag_tvalid => m_axis_mag_tvalid,
      m_axis_mag_tready => m_axis_mag_tready,
      m_axis_mag_tdata  => m_axis_mag_tdata,
      m_axis_mag_tuser  => m_axis_mag_tuser,
      m_axis_mag_tlast  => m_axis_mag_tlast,

      m_axis_dir_tvalid => m_axis_dir_tvalid,
      m_axis_dir_tready => m_axis_dir_tready,
      m_axis_dir_tdata  => m_axis_dir_tdata,
      m_axis_dir_tuser  => m_axis_dir_tuser,
      m_axis_dir_tlast  => m_axis_dir_tlast
    );

end architecture tb;
