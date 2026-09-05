library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.axi_stream_pkg.all;

library osvvm;
use osvvm.RandomPkg.all;

-- TDD testbench for modules/axi_stream_join/src/axi_stream_join.vhd, written
-- before the two --@ markers in that file are resolved (per the project's
-- TDD policy in shared/Vunit.md #15). Uses VUnit's raw axi_stream_master/
-- axi_stream_slave verification components directly (not the hdl-modules
-- bfm.* wrappers), because bfm.axi_stream_master/slave require
-- user_width mod 8 = 0 and this module's tuser is 2 bits -- see
-- shared/Vunit.md #12 and vhtestgen/SKILL.md.
entity tb_axi_stream_join is
  generic (
    -- Set per VUnit test config in module_axi_stream_join.py: 0 for the
    -- full-throughput test, nonzero (randomized backpressure) otherwise.
    stall_probability_percent : natural;
    runner_cfg : string
  );
end entity tb_axi_stream_join;

architecture tb of tb_axi_stream_join is

  constant data_width_a : positive := 8;
  constant data_width_b : positive := 6;
  constant user_width   : positive := 2;

  constant clock_period : time := 10 ns;

  signal clk : std_logic := '0';

  signal s_axis_a_tvalid, s_axis_a_tready, s_axis_a_tlast : std_logic;
  signal s_axis_a_tdata : std_logic_vector(data_width_a - 1 downto 0);
  signal s_axis_a_tuser : std_logic_vector(user_width - 1 downto 0);

  signal s_axis_b_tvalid, s_axis_b_tready, s_axis_b_tlast : std_logic;
  signal s_axis_b_tdata : std_logic_vector(data_width_b - 1 downto 0);
  signal s_axis_b_tuser : std_logic_vector(user_width - 1 downto 0);

  signal m_axis_tvalid, m_axis_tready, m_axis_tlast : std_logic;
  signal m_axis_tdata : std_logic_vector(data_width_a + data_width_b - 1 downto 0);
  signal m_axis_tuser : std_logic_vector(user_width - 1 downto 0);

  constant stall_config : stall_config_t := new_stall_config(
    stall_probability => real(stall_probability_percent) / 100.0,
    min_stall_cycles   => 1,
    max_stall_cycles   => 4
  );

  constant axi_master_a : axi_stream_master_t := new_axi_stream_master(
    data_length  => data_width_a,
    user_length  => user_width,
    stall_config => stall_config,
    logger       => get_logger("axi_master_a")
  );
  constant axi_master_b : axi_stream_master_t := new_axi_stream_master(
    data_length  => data_width_b,
    user_length  => user_width,
    stall_config => stall_config,
    logger       => get_logger("axi_master_b")
  );
  constant axi_slave_result : axi_stream_slave_t := new_axi_stream_slave(
    data_length  => data_width_a + data_width_b,
    user_length  => user_width,
    stall_config => stall_config,
    logger       => get_logger("axi_slave_result")
  );

begin

  test_runner_watchdog(runner, 2 ms);
  clk <= not clk after clock_period / 2;


  ------------------------------------------------------------------------------
  main : process
    variable rnd : RandomPType;

    -- Pushes 'num_words' beats to both s_axis_a and s_axis_b (matching tlast
    -- so the two lanes stay frame-synchronized, per the module's own
    -- requirement) with independently randomized per-lane border bits, and
    -- checks the joined result: data concatenation, tuser(1) = OR of the two
    -- lanes' border bits, tuser(0)/tlast taken from lane A.
    procedure run_join_test(
      num_words        : positive;
      check_throughput : boolean := false
    ) is
      variable a_word, b_word     : natural;
      variable a_border, b_border : std_logic;
      variable sof, is_last       : std_logic;
      variable expected_data      : std_logic_vector(
        data_width_a + data_width_b - 1 downto 0
      );
      variable expected_user : std_logic_vector(user_width - 1 downto 0);
      variable start_time    : time;
    begin
      start_time := now;

      for word_idx in 0 to num_words - 1 loop
        a_word   := rnd.RandInt(0, 2 ** data_width_a - 1);
        b_word   := rnd.RandInt(0, 2 ** data_width_b - 1);
        -- OSVVM's RandSlv(Size) returns a (1 to Size)-ranged vector, not
        -- (Size - 1 downto 0), so index with (1) not (0).
        a_border := rnd.RandSlv(1)(1);
        b_border := rnd.RandSlv(1)(1);

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

        expected_data := (
          std_logic_vector(to_unsigned(a_word, data_width_a))
          & std_logic_vector(to_unsigned(b_word, data_width_b))
        );
        -- Leftmost element of the actual maps to the formal's high (MSB)
        -- index regardless of the literal's own ascending/descending index
        -- numbering, so "border & sof" correctly lands border at tuser(1)
        -- and sof at tuser(0) (matching the DUT's tuser(1 downto 0)).
        expected_user := (a_border or b_border) & sof;

        push_axi_stream(
          net        => net,
          axi_stream => axi_master_a,
          tdata      => std_logic_vector(to_unsigned(a_word, data_width_a)),
          tlast      => is_last,
          tuser      => a_border & sof
        );
        push_axi_stream(
          net        => net,
          axi_stream => axi_master_b,
          tdata      => std_logic_vector(to_unsigned(b_word, data_width_b)),
          tlast      => is_last,
          tuser      => b_border & sof
        );

        check_axi_stream(
          net        => net,
          axi_stream => axi_slave_result,
          expected   => expected_data,
          tlast      => is_last,
          tuser      => expected_user,
          msg        => "word_idx=" & to_string(word_idx)
        );
      end loop;

      if check_throughput then
        -- Combinational rendezvous, zero stall on all three sides: must
        -- sustain one output beat per clock cycle, i.e. finish in
        -- essentially num_words cycles (small margin for the initial
        -- reset/startup latency only).
        check_relation(
          (now - start_time) < (num_words + 5) * clock_period,
          "join did not sustain full throughput at zero stall"
        );
      end if;
    end procedure;

    variable a_border, b_border, sof, is_last : std_logic;

  begin
    test_runner_setup(runner, runner_cfg);
    rnd.InitSeed(get_string_seed(runner_cfg));

    if run("test_random_data") then
      run_join_test(num_words => 300);

    elsif run("test_full_throughput") then
      run_join_test(num_words => 500, check_throughput => true);

    elsif run("test_border_combinations") then
      -- Directed: exercise every combination of the independent per-lane
      -- border bit within one 4-word packet (both lanes always agree on
      -- sof/tlast, since that agreement is this module's own precondition,
      -- checked separately by its fork-desync assertion -- not exercised
      -- here, see doc/axi_stream_join.md "Verification notes").
      for combo in 0 to 3 loop
        a_border := to_unsigned(combo, 2)(0);
        b_border := to_unsigned(combo, 2)(1);

        if combo = 0 then
          sof := '1';
        else
          sof := '0';
        end if;

        if combo = 3 then
          is_last := '1';
        else
          is_last := '0';
        end if;

        push_axi_stream(
          net        => net,
          axi_stream => axi_master_a,
          tdata      => std_logic_vector(to_unsigned(combo, data_width_a)),
          tlast      => is_last,
          tuser      => a_border & sof
        );
        push_axi_stream(
          net        => net,
          axi_stream => axi_master_b,
          tdata      => std_logic_vector(to_unsigned(combo, data_width_b)),
          tlast      => is_last,
          tuser      => b_border & sof
        );
        check_axi_stream(
          net        => net,
          axi_stream => axi_slave_result,
          expected   => std_logic_vector(to_unsigned(combo, data_width_a))
                        & std_logic_vector(to_unsigned(combo, data_width_b)),
          tlast      => is_last,
          tuser      => (a_border or b_border) & sof,
          msg        => "combo=" & to_string(combo)
        );
      end loop;

    end if;

    test_runner_cleanup(runner);
  end process;


  ------------------------------------------------------------------------------
  axi_stream_master_a_inst : entity vunit_lib.axi_stream_master
    generic map (
      master => axi_master_a
    )
    port map (
      aclk   => clk,
      tvalid => s_axis_a_tvalid,
      tready => s_axis_a_tready,
      tdata  => s_axis_a_tdata,
      tlast  => s_axis_a_tlast,
      tuser  => s_axis_a_tuser
    );

  axi_stream_master_b_inst : entity vunit_lib.axi_stream_master
    generic map (
      master => axi_master_b
    )
    port map (
      aclk   => clk,
      tvalid => s_axis_b_tvalid,
      tready => s_axis_b_tready,
      tdata  => s_axis_b_tdata,
      tlast  => s_axis_b_tlast,
      tuser  => s_axis_b_tuser
    );

  axi_stream_slave_result_inst : entity vunit_lib.axi_stream_slave
    generic map (
      slave => axi_slave_result
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
  dut : entity work.axi_stream_join
    generic map (
      data_width_a => data_width_a,
      data_width_b => data_width_b
    )
    port map (
      clk => clk,

      s_axis_a_tvalid => s_axis_a_tvalid,
      s_axis_a_tready => s_axis_a_tready,
      s_axis_a_tdata  => s_axis_a_tdata,
      s_axis_a_tuser  => s_axis_a_tuser,
      s_axis_a_tlast  => s_axis_a_tlast,

      s_axis_b_tvalid => s_axis_b_tvalid,
      s_axis_b_tready => s_axis_b_tready,
      s_axis_b_tdata  => s_axis_b_tdata,
      s_axis_b_tuser  => s_axis_b_tuser,
      s_axis_b_tlast  => s_axis_b_tlast,

      m_axis_tvalid => m_axis_tvalid,
      m_axis_tready => m_axis_tready,
      m_axis_tdata  => m_axis_tdata,
      m_axis_tuser  => m_axis_tuser,
      m_axis_tlast  => m_axis_tlast
    );

end architecture tb;
