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

-- TDD testbench for modules/canny/src/canny_gaussian3x3.vhd,
-- written before the --@ markers in that file are resolved (per the
-- project's TDD policy in shared/Vunit.md #15). Uses VUnit's raw
-- axi_stream_master/axi_stream_slave verification components directly (not
-- the hdl-modules bfm.* wrappers), because bfm.axi_stream_master/slave
-- require user_width mod 8 = 0 and this module's tuser is 2 bits -- see
-- shared/Vunit.md #12 and vhtestgen/SKILL.md (same precedent as
-- tb_canny_threshold.vhd).
entity tb_canny_gaussian3x3 is
  generic (
    -- Set per VUnit test config in module_canny.py: 0 for the
    -- full-throughput test, nonzero (randomized backpressure) otherwise.
    stall_probability_percent : natural;
    runner_cfg : string
  );
end entity tb_canny_gaussian3x3;

architecture tb of tb_canny_gaussian3x3 is

  constant tap_width  : positive := 8;
  constant data_width : positive := 9 * tap_width;
  constant user_width : positive := 2;

  constant clk_period : time := 10 ns;

  signal clk : std_logic := '0';

  signal s_axis_tvalid, s_axis_tready, s_axis_tlast : std_logic;
  signal s_axis_tdata : std_logic_vector(data_width - 1 downto 0);
  signal s_axis_tuser : std_logic_vector(user_width - 1 downto 0);

  signal m_axis_tvalid, m_axis_tready, m_axis_tlast : std_logic;
  signal m_axis_tdata : std_logic_vector(tap_width - 1 downto 0);
  signal m_axis_tuser : std_logic_vector(user_width - 1 downto 0);

  constant stall_config : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master_in : axi_stream_master_t := new_axi_stream_master(
    data_length  => data_width,
    user_length  => user_width,
    stall_config => stall_config,
    logger       => get_logger("axi_master_in")
  );
  constant axi_slave_out : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => tap_width,
    user_length  => user_width,
    stall_config => stall_config,
    logger       => get_logger("axi_slave_out")
  );

  -- Independently re-derived from the requirement (not copy-pasted from the
  -- RTL): weighted sum with integer weights [1,2,1;2,4,2;1,2,1] (sum=16),
  -- exact power-of-two truncating divide via right-shift; border ('1')
  -- forces the result to 0.
  function expected_gaussian(
    tl, tm, tr, ml, mm, mr, bl, bm, br : natural;
    border : std_logic
  ) return std_logic_vector is
    variable weighted_sum : natural;
  begin
    if border = '1' then
      return std_logic_vector(to_unsigned(0, tap_width));
    end if;

    weighted_sum := tl + 2 * tm + tr
                  + 2 * ml + 4 * mm + 2 * mr
                  + bl + 2 * bm + br;

    return std_logic_vector(to_unsigned((weighted_sum / 16) mod 256, tap_width));
  end function;

begin

  test_runner_watchdog(runner, 2 ms);
  clk <= not clk after clk_period / 2;


  ------------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;

    -- Packs the 9 taps row-major MSB-to-LSB per the requirement's port
    -- table, pushes one beat and checks the expected smoothed result
    -- against expected_gaussian above.
    procedure push_and_check(
      tl, tm, tr, ml, mm, mr, bl, bm, br : natural;
      border  : std_logic;
      sof     : std_logic;
      is_last : std_logic;
      msg     : string := ""
    ) is
      variable tdata : std_logic_vector(data_width - 1 downto 0);
    begin
      tdata := std_logic_vector(to_unsigned(tl, tap_width))
             & std_logic_vector(to_unsigned(tm, tap_width))
             & std_logic_vector(to_unsigned(tr, tap_width))
             & std_logic_vector(to_unsigned(ml, tap_width))
             & std_logic_vector(to_unsigned(mm, tap_width))
             & std_logic_vector(to_unsigned(mr, tap_width))
             & std_logic_vector(to_unsigned(bl, tap_width))
             & std_logic_vector(to_unsigned(bm, tap_width))
             & std_logic_vector(to_unsigned(br, tap_width));

      push_axi_stream(
        net        => net,
        axi_stream => axi_master_in,
        tdata      => tdata,
        tlast      => is_last,
        tuser      => border & sof
      );
      check_axi_stream(
        net        => net,
        axi_stream => axi_slave_out,
        expected   => expected_gaussian(tl, tm, tr, ml, mm, mr, bl, bm, br, border),
        tlast      => is_last,
        tuser      => border & sof,
        msg        => msg,
        blocking   => false
      );
    end procedure;

    variable start_time : time;
    variable t : integer_vector(0 to 8);
    variable border : std_logic;
    variable sof, is_last : std_logic;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    if run("test_random_data") then
      -- Random taps across the full 0..255 range, random border bit.
      for word_idx in 0 to 299 loop
        for i in 0 to 8 loop
          t(i) := rnd.RandInt(0, 2 ** tap_width - 1);
        end loop;
        -- OSVVM's RandSlv(Size) returns a (1 to Size)-ranged vector, not
        -- (Size - 1 downto 0), so index with (1) not (0).
        border := rnd.RandSlv(1)(1);

        if word_idx = 0 then
          sof := '1';
        else
          sof := '0';
        end if;

        if word_idx = 299 then
          is_last := '1';
        else
          is_last := '0';
        end if;

        push_and_check(
          t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8),
          border, sof, is_last, "word_idx=" & to_string(word_idx)
        );
      end loop;

    elsif run("test_border_forces_zero") then
      -- Directed: taps that would otherwise sum to a large nonzero result,
      -- with border='1', expecting 0 regardless.
      push_and_check(255, 255, 255, 255, 255, 255, 255, 255, 255, '1', '1', '0', "border forces zero (max taps)");
      push_and_check(100, 50, 20, 10, 5, 5, 1, 1, 1, '1', '0', '0', "border forces zero (mixed taps)");
      push_and_check(0, 0, 0, 0, 0, 0, 0, 0, 0, '1', '0', '1', "border forces zero (zero taps)");

    elsif run("test_full_throughput") then
      -- One-cycle-latency elastic stage, zero stall on both sides: must
      -- sustain one output beat per clock cycle, i.e. finish in
      -- essentially num_words cycles (small margin for the initial
      -- reset/startup/pipeline-fill latency only).
      start_time := now;

      for word_idx in 0 to 499 loop
        for i in 0 to 8 loop
          t(i) := rnd.RandInt(0, 2 ** tap_width - 1);
        end loop;
        border := rnd.RandSlv(1)(1);

        if word_idx = 0 then
          sof := '1';
        else
          sof := '0';
        end if;

        if word_idx = 499 then
          is_last := '1';
        else
          is_last := '0';
        end if;

        push_and_check(
          t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8),
          border, sof, is_last, "word_idx=" & to_string(word_idx)
        );
      end loop;

      check_relation(
        (now - start_time) < (500 + 10) * clk_period,
        "canny_gaussian3x3 did not sustain full throughput at zero stall"
      );

    end if;

    -- push_axi_stream and check_axi_stream(blocking => false) only enqueue
    -- messages to the VC actors (com's send() does not block for an
    -- unbounded inbox) -- without this, test_runner_cleanup's core_pkg.stop
    -- would halt the simulation before the VCs actually drive/sample any
    -- bus cycles, silently "passing" with zero real checks performed. Must
    -- wait for both VCs to actually finish driving/checking every queued
    -- beat first.
    wait_until_idle(net, as_sync(axi_master_in));
    wait_until_idle(net, as_sync(axi_slave_out));

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
  axi_stream_master_inst : entity vunit_lib.axi_stream_master
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

  axi_stream_slave_inst : entity vunit_lib.axi_stream_slave
    generic map (
      slave => axi_slave_out
    )
    port map (
      aclk   => clk,
      tvalid => m_axis_tvalid,
      tready => m_axis_tready,
      tdata  => m_axis_tdata,
      tlast  => m_axis_tlast,
      tuser  => m_axis_tuser
    );


  ------------------------------------------------------------------------------
  dut : entity work.canny_gaussian3x3
    port map (
      clk => clk,

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
