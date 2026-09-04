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

-- TDD testbench for modules/canny_hysteresis/src/canny_hysteresis.vhd, written
-- before the --@ markers in that file are resolved (per shared/Vunit.md #15).
-- Uses VUnit's raw axi_stream_master/axi_stream_slave verification
-- components directly (not the hdl-modules bfm.* wrappers): both this
-- module's s_axis_tuser (2 bits) and m_axis_tuser (1 bit) violate
-- bfm.axi_stream_master/slave's user_width mod 8 = 0 assertion, so raw
-- VUnit VCs are used consistently on both sides (not a mix) -- see
-- shared/Vunit.md #12 and modules/canny_hysteresis/doc/canny_hysteresis_proposal.md
-- "Verification plan".
entity tb_canny_hysteresis is
  generic (
    -- Set per VUnit test config in module_canny_hysteresis.py: 0 for the
    -- full-throughput test, nonzero (randomized backpressure) otherwise.
    stall_probability_percent : natural;
    runner_cfg : string
  );
end entity tb_canny_hysteresis;

architecture tb of tb_canny_hysteresis is

  constant c_s_data_width : positive := 18;
  constant c_s_user_width : positive := 2;
  constant c_m_data_width : positive := 8;
  constant c_m_user_width : positive := 1;

  constant c_clk_period : time := 10 ns;

  signal clk   : std_logic := '0';
  signal rst_n : std_logic := '0';

  signal s_axis_tvalid, s_axis_tready, s_axis_tlast : std_logic;
  signal s_axis_tdata : std_logic_vector(c_s_data_width - 1 downto 0);
  signal s_axis_tuser : std_logic_vector(c_s_user_width - 1 downto 0);

  signal m_axis_tvalid, m_axis_tready, m_axis_tlast : std_logic;
  signal m_axis_tdata : std_logic_vector(c_m_data_width - 1 downto 0);
  signal m_axis_tuser : std_logic_vector(c_m_user_width - 1 downto 0);

  constant c_stall_config : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master : axi_stream_master_t := new_axi_stream_master(
    data_length  => c_s_data_width,
    user_length  => c_s_user_width,
    stall_config => c_stall_config,
    logger       => get_logger("axi_master")
  );
  constant axi_slave : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => c_m_data_width,
    user_length  => c_m_user_width,
    stall_config => c_stall_config,
    logger       => get_logger("axi_slave")
  );

  -- Classification codes (2 bits/tap), matching the DUT/requirement.
  constant c_strong : std_logic_vector(1 downto 0) := "10";
  constant c_weak   : std_logic_vector(1 downto 0) := "01";
  constant c_none   : std_logic_vector(1 downto 0) := "00";

  -- 9 taps, index order matching s_axis_tdata's row-major packing:
  -- 0=w_tl, 1=w_tc, 2=w_tr, 3=w_ml, 4=w_mm (center), 5=w_mr, 6=w_bl, 7=w_bc, 8=w_br.
  type tap_array_t is array (0 to 8) of std_logic_vector(1 downto 0);
  constant c_center_idx : natural := 4;

  function pack_window(taps : tap_array_t) return std_logic_vector is
    variable result : std_logic_vector(c_s_data_width - 1 downto 0);
  begin
    for tap_idx in 0 to 8 loop
      result(17 - 2 * tap_idx downto 16 - 2 * tap_idx) := taps(tap_idx);
    end loop;
    return result;
  end function;

  -- Golden model: matches the requirement's hysteresis rule exactly
  -- (also matches the DUT's compute_edge function once implemented).
  function expected_edge(taps : tap_array_t; border : std_logic) return std_logic is
    variable any_neighbor_strong : boolean := false;
    variable e : std_logic;
  begin
    for tap_idx in 0 to 8 loop
      if tap_idx /= c_center_idx and taps(tap_idx) = c_strong then
        any_neighbor_strong := true;
      end if;
    end loop;

    if taps(c_center_idx) = c_strong then
      e := '1';
    elsif taps(c_center_idx) = c_weak and any_neighbor_strong then
      e := '1';
    else
      e := '0';
    end if;

    if border = '1' then
      e := '0';
    end if;

    return e;
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

    -- Pushes one beat with the given 3x3 classification window, border and
    -- SOF bits, and queues the corresponding expected m_axis check
    -- (non-blocking, per shared/Vunit.md #12).
    procedure run_beat(
      taps    : tap_array_t;
      border  : std_logic;
      sof     : std_logic;
      is_last : std_logic;
      msg     : string := ""
    ) is
      variable expected_data : std_logic_vector(c_m_data_width - 1 downto 0);
      variable expected_user : std_logic_vector(c_m_user_width - 1 downto 0);
    begin
      expected_data := "0000000" & expected_edge(taps, border);
      expected_user(0) := sof;

      push_axi_stream(
        net        => net,
        axi_stream => axi_master,
        tdata      => pack_window(taps),
        tlast      => is_last,
        tuser      => border & sof
      );
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

    -- Pushes 'num_words' beats of fully random windows/border/SOF, checking
    -- each against the golden model above.
    procedure run_random_test(num_words : positive) is
      variable taps    : tap_array_t;
      variable border, sof, is_last : std_logic;
    begin
      for word_idx in 0 to num_words - 1 loop
        for tap_idx in 0 to 8 loop
          taps(tap_idx) := std_logic_vector(to_unsigned(rnd.RandInt(0, 3), 2));
        end loop;
        -- OSVVM's RandSlv(Size) returns a (1 to Size)-ranged vector, not
        -- (Size - 1 downto 0), so index with (1) not (0).
        border := rnd.RandSlv(1)(1);

        if word_idx = 0 then
          sof := '1';
        else
          sof := '0';
        end if;

        if word_idx = num_words - 1 then
          is_last := '1';
        else
          is_last := '0';
        end if;

        run_beat(
          taps    => taps,
          border  => border,
          sof     => sof,
          is_last => is_last,
          msg     => "word_idx=" & to_string(word_idx)
        );
      end loop;
    end procedure;

    variable start_time : time;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    wait until rst_n = '1' and rising_edge(clk);

    if run("test_hysteresis_combinations") then
      -- a) center=weak, a neighbor strong => edge
      run_beat(
        taps    => (c_strong, c_none, c_none, c_none, c_weak, c_none, c_none, c_none, c_none),
        border  => '0', sof => '1', is_last => '0',
        msg     => "weak center, strong neighbor -> edge"
      );

      -- b) center=weak, no neighbor strong => no edge
      run_beat(
        taps    => (c_none, c_weak, c_none, c_none, c_weak, c_none, c_weak, c_none, c_none),
        border  => '0', sof => '0', is_last => '0',
        msg     => "weak center, no strong neighbor -> no edge"
      );

      -- c) center=strong => always edge, regardless of neighbors
      run_beat(
        taps    => (c_none, c_weak, c_none, c_none, c_strong, c_none, c_weak, c_none, c_none),
        border  => '0', sof => '0', is_last => '0',
        msg     => "strong center -> always edge"
      );

      -- d) center=none => never edge, regardless of neighbors
      run_beat(
        taps    => (c_strong, c_none, c_none, c_none, c_none, c_none, c_strong, c_none, c_none),
        border  => '0', sof => '0', is_last => '1',
        msg     => "none center -> never edge"
      );

    elsif run("test_border_forces_edge_zero") then
      -- Window comparison alone would give edge='1' (strong center), but
      -- border='1' must force edge='0'. SOF must still pass through
      -- unchanged on the same beat.
      run_beat(
        taps    => (c_strong, c_strong, c_strong, c_strong, c_strong, c_strong, c_strong, c_strong,
                     c_strong),
        border  => '1', sof => '1', is_last => '1',
        msg     => "border forces edge=0, sof still passes through"
      );

    elsif run("test_random_data") then
      run_random_test(num_words => 300);

    elsif run("test_full_throughput") then
      start_time := now;
      run_random_test(num_words => 500);

      -- One registered pipeline stage (common.handshake_pipeline, full
      -- skid-buffer mode), zero stall on both sides: must sustain one
      -- output beat per clock cycle, i.e. finish in essentially num_words
      -- cycles (small margin for the initial reset/startup latency and the
      -- one-cycle pipeline latency only).
      check_relation(
        (now - start_time) < (500 + 5) * c_clk_period,
        "hysteresis stage did not sustain full throughput at zero stall"
      );

    end if;

    -- push_axi_stream / non-blocking check_axi_stream only enqueue messages
    -- on the VCs' internal queues; without this, test_runner_cleanup would
    -- end the simulation immediately, before any beat is actually driven or
    -- checked (a stubbed/broken DUT would then falsely "pass").
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


  ------------------------------------------------------------------------------
  dut : entity work.canny_hysteresis
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
