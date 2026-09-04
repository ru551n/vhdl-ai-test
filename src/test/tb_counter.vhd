library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;

entity tb_counter is
  generic (
    runner_cfg : string
  );
end entity tb_counter;

architecture tb of tb_counter is
  constant width_c : positive := 8;

  signal clk    : std_logic := '0';
  signal rst    : std_logic := '0';
  signal enable : std_logic := '0';
  signal count  : unsigned(width_c - 1 downto 0);
begin

  clk <= not clk after 5 ns;

  dut : entity work.counter
    generic map (
      width => width_c
    )
    port map (
      clk    => clk,
      rst    => rst,
      enable => enable,
      count  => count
    );

  main : process is
  begin
    test_runner_setup(runner, runner_cfg);

    while test_suite loop
      if run("test_reset_clears_counter") then
        enable <= '0';
        rst    <= '1';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        rst <= '0';
        wait for 1 ns;
        check_equal(count, 0, "count not zero after reset");

      elsif run("test_counts_up_when_enabled") then
        rst    <= '1';
        wait until rising_edge(clk);
        rst    <= '0';
        enable <= '1';
        for i in 1 to 5 loop
          wait until rising_edge(clk);
        end loop;
        wait for 1 ns;
        check_equal(count, 5, "count did not reach expected value");

      elsif run("test_holds_when_disabled") then
        rst    <= '1';
        wait until rising_edge(clk);
        rst    <= '0';
        enable <= '0';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for 1 ns;
        check_equal(count, 0, "count changed while disabled");
      end if;
    end loop;

    test_runner_cleanup(runner);
  end process main;

  test_runner_watchdog(runner, 10 us);

end architecture tb;
