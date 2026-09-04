library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.axi_stream_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

-- TDD testbench for modules/canny_threshold/src/canny_threshold.vhd, written
-- before the --@ marker in that file is resolved (per the project's TDD
-- policy in shared/Vunit.md #15). Uses VUnit's raw axi_stream_master/
-- axi_stream_slave verification components directly (not the hdl-modules
-- bfm.* wrappers), because bfm.axi_stream_master/slave require
-- user_width mod 8 = 0 and this module's tuser is 2 bits -- see
-- shared/Vunit.md #12 and vhtestgen/SKILL.md.
--
-- IMPORTANT: does NOT use vunit_lib's check_axi_stream for the output side.
-- check_axi_stream's/axi_stream_slave.vhd's TDATA mismatch detection is
-- gated by "for idx in tkeep'range loop ... mismatch := tdata(...) /=
-- expected(...)"; tkeep's width is data_length/8, so for m_axis_tdata
-- (data_length => 2, not a multiple of 8) tkeep is a null-range vector,
-- the loop body never runs, "mismatch" is never set true, and TDATA is
-- silently never checked. This is a real VUnit limitation for any
-- data_length not a multiple of 8 (same root cause as the bfm.* mod-8
-- restriction), not specific to this module. Worked around below with a
-- manual expected-value queue plus pop_axi_stream + check_equal in a
-- separate checker process (TLAST/TUSER are checked unconditionally by
-- check_axi_stream and would have been fine, but are also folded into
-- this same manual check for consistency).
entity tb_canny_threshold is
  generic (
    -- Set per VUnit test config in module_canny_threshold.py: 0 for the
    -- full-throughput test, nonzero (randomized backpressure) otherwise.
    stall_probability_percent : natural;
    runner_cfg : string
  );
end entity tb_canny_threshold;

architecture tb of tb_canny_threshold is

  constant c_data_width : positive := 11;
  constant c_user_width : positive := 2;

  constant c_thresh_low  : natural := 100;
  constant c_thresh_high : natural := 200;

  constant c_clk_period : time := 10 ns;

  signal clk   : std_logic := '0';
  signal rst_n : std_logic := '0';

  signal s_axis_tvalid, s_axis_tready, s_axis_tlast : std_logic;
  signal s_axis_tdata : std_logic_vector(c_data_width - 1 downto 0);
  signal s_axis_tuser : std_logic_vector(c_user_width - 1 downto 0);

  signal m_axis_tvalid, m_axis_tready, m_axis_tlast : std_logic;
  signal m_axis_tdata : std_logic_vector(1 downto 0);
  signal m_axis_tuser : std_logic_vector(c_user_width - 1 downto 0);

  constant c_stall_config : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master_in : axi_stream_master_t := new_axi_stream_master(
    data_length  => c_data_width,
    user_length  => c_user_width,
    stall_config => c_stall_config,
    logger       => get_logger("axi_master_in")
  );
  constant axi_slave_out : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => 2,
    user_length  => c_user_width,
    stall_config => c_stall_config,
    logger       => get_logger("axi_slave_out")
  );

  -- Expected (tdata & tlast & tuser) queued by the main process in push
  -- order, drained by checker_proc below against the actual DUT output --
  -- see the top-of-file comment on why check_axi_stream can't be used here.
  constant expected_q : queue_t := new_queue;

  -- Bumped by checker_proc after each beat is checked; the main process
  -- waits for this to catch up with the number of beats it pushed before
  -- calling test_runner_cleanup, so no check is dropped at the end of a run.
  signal num_checked : natural := 0;

  -- Independently re-derived from the requirement (not copy-pasted from the
  -- RTL): border forces "00" unconditionally; else "10" (strong) at/above
  -- g_thresh_high; "01" (weak) at/above g_thresh_low; else "00" (none).
  function expected_classify(
    magnitude : natural;
    border    : std_logic
  ) return std_logic_vector is
  begin
    if border = '1' then
      return "00";
    elsif magnitude >= c_thresh_high then
      return "10";
    elsif magnitude >= c_thresh_low then
      return "01";
    else
      return "00";
    end if;
  end function;

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
    variable num_pushed  : natural := 0;

    -- Pushes one beat with the given magnitude/border/sof/tlast and queues
    -- the expected classification (against expected_classify above) for
    -- checker_proc to verify once the corresponding output beat appears.
    procedure push_and_check(
      magnitude : natural;
      border    : std_logic;
      sof       : std_logic;
      is_last   : std_logic;
      msg       : string := ""
    ) is
    begin
      push_axi_stream(
        net        => net,
        axi_stream => axi_master_in,
        tdata      => std_logic_vector(to_unsigned(magnitude, c_data_width)),
        tlast      => is_last,
        tuser      => border & sof
      );

      push_std_ulogic_vector(expected_q, expected_classify(magnitude, border));
      push_std_ulogic(expected_q, is_last);
      push_std_ulogic_vector(expected_q, border & sof);

      num_pushed := num_pushed + 1;
    end procedure;

    variable start_time : time;
    variable magnitude   : natural;
    variable border      : std_logic;
    variable sof, is_last : std_logic;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    wait until rst_n = '1' and rising_edge(clk);

    if run("test_boundary_values") then
      -- Directed: exactly g_thresh_low-1, g_thresh_low, g_thresh_high-1,
      -- g_thresh_high, non-border.
      push_and_check(c_thresh_low - 1, '0', '1', '0', "thresh_low-1");
      push_and_check(c_thresh_low, '0', '0', '0', "thresh_low");
      push_and_check(c_thresh_high - 1, '0', '0', '0', "thresh_high-1");
      push_and_check(c_thresh_high, '0', '0', '1', "thresh_high");

    elsif run("test_random_data") then
      -- Random magnitude across the full 0..2047 (11-bit) input range,
      -- random border bit.
      for word_idx in 0 to 299 loop
        magnitude := rnd.RandInt(0, 2 ** c_data_width - 1);
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

        push_and_check(magnitude, border, sof, is_last, "word_idx=" & to_string(word_idx));
      end loop;

    elsif run("test_border_forces_none") then
      -- Directed: magnitudes that would otherwise classify as weak/strong,
      -- with border='1', expecting "00" regardless.
      push_and_check(c_thresh_low + 10, '1', '1', '0', "border forces none (weak magnitude)");
      push_and_check(c_thresh_high + 10, '1', '0', '0', "border forces none (strong magnitude)");
      push_and_check(0, '1', '0', '1', "border forces none (zero magnitude)");

    elsif run("test_full_throughput") then
      -- One-cycle-latency elastic stage, zero stall on both sides: must
      -- sustain one output beat per clock cycle, i.e. finish in
      -- essentially num_words cycles (small margin for the initial
      -- reset/startup/pipeline-fill latency only).
      start_time := now;

      for word_idx in 0 to 499 loop
        magnitude := rnd.RandInt(0, 2 ** c_data_width - 1);
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

        push_and_check(magnitude, border, sof, is_last, "word_idx=" & to_string(word_idx));
      end loop;

      check_relation(
        (now - start_time) < (500 + 10) * c_clk_period,
        "canny_threshold did not sustain full throughput at zero stall"
      );

    end if;

    -- Let checker_proc drain expected_q against the actual DUT output
    -- before ending the simulation, so the last beats' checks aren't lost.
    wait until num_checked = num_pushed;

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
  -- Pops each output beat as it arrives and checks it against the expected
  -- value queued by push_and_check above. Runs concurrently with the main
  -- process (not interleaved push-then-blocking-pop in the same process),
  -- so backpressure-free pushing/checking can still sustain full throughput.
  checker_proc : process
    variable got_tdata : std_logic_vector(1 downto 0);
    variable got_tlast : std_logic;
    variable got_tkeep, got_tstrb : std_logic_vector(data_length(axi_slave_out) / 8 - 1 downto 0);
    variable got_tid   : std_logic_vector(id_length(axi_slave_out) - 1 downto 0);
    variable got_tdest : std_logic_vector(dest_length(axi_slave_out) - 1 downto 0);
    variable got_tuser : std_logic_vector(c_user_width - 1 downto 0);

    variable expected_tdata : std_logic_vector(1 downto 0);
    variable expected_tlast : std_logic;
    variable expected_tuser : std_logic_vector(c_user_width - 1 downto 0);
  begin
    loop
      pop_axi_stream(
        net        => net,
        axi_stream => axi_slave_out,
        tdata      => got_tdata,
        tlast      => got_tlast,
        tkeep      => got_tkeep,
        tstrb      => got_tstrb,
        tid        => got_tid,
        tdest      => got_tdest,
        tuser      => got_tuser
      );

      expected_tdata := pop_std_ulogic_vector(expected_q);
      expected_tlast := pop_std_ulogic(expected_q);
      expected_tuser := pop_std_ulogic_vector(expected_q);

      check_equal(got_tdata, expected_tdata, "TDATA mismatch, beat " & to_string(num_checked));
      check_equal(got_tlast, expected_tlast, "TLAST mismatch, beat " & to_string(num_checked));
      check_equal(got_tuser, expected_tuser, "TUSER mismatch, beat " & to_string(num_checked));

      num_checked <= num_checked + 1;
    end loop;
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
  dut : entity work.canny_threshold
    generic map (
      g_thresh_low  => c_thresh_low,
      g_thresh_high => c_thresh_high
    )
    port map (
      clk   => clk,
      rst_n => rst_n,

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
