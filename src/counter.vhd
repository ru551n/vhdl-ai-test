library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity counter is
  generic (
    width : positive := 8
  );
  port (
    clk    : in  std_logic;
    rst    : in  std_logic;
    enable : in  std_logic;
    count  : out unsigned(width - 1 downto 0)
  );
end entity counter;

architecture rtl of counter is
  signal count_int : unsigned(width - 1 downto 0) := (others => '0');
begin

  main : process (clk) is
  begin
    if rising_edge(clk) then
      if rst = '1' then
        count_int <= (others => '0');
      elsif enable = '1' then
        count_int <= count_int + 1;
      end if;
    end if;
  end process main;

  count <= count_int;

end architecture rtl;
