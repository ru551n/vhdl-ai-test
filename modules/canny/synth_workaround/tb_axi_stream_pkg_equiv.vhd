library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library orig_pkg;
library patched_pkg;

entity tb_axi_stream_pkg_equiv is
end entity;

architecture tb of tb_axi_stream_pkg_equiv is
begin

  main : process
    variable o_data, p_data : natural;

    variable o_m2s : orig_pkg.axi_stream_pkg.axi_stream_m2s_t;
    variable p_m2s : patched_pkg.axi_stream_pkg.axi_stream_m2s_t;

    variable data_field : std_ulogic_vector(127 downto 0);
    variable user_field : std_ulogic_vector(15 downto 0);
    variable last_bit   : std_ulogic;

    variable o_slv, p_slv : std_ulogic_vector(
      orig_pkg.axi_stream_pkg.axi_stream_data_sz +
      orig_pkg.axi_stream_pkg.axi_stream_user_sz downto 0
    );

    variable o_back, p_back : orig_pkg.axi_stream_pkg.axi_stream_m2s_t;
    variable p_back2 : patched_pkg.axi_stream_pkg.axi_stream_m2s_t;

    variable mismatches : natural := 0;
    variable checks : natural := 0;

    type width_arr_t is array (natural range <>) of positive;
    constant data_widths : width_arr_t := (1, 2, 8, 11, 16, 128);

    -- One "pattern generator" per iteration: alternating, all-ones,
    -- all-zeros, incrementing, walking-1 -- driven off the loop index so
    -- every (data_width, user_width) combo gets varied content, not just
    -- one fixed value.
    function pattern(seed : natural; width : positive) return std_ulogic_vector is
      variable result : std_ulogic_vector(width - 1 downto 0);
    begin
      case seed mod 5 is
        when 0 => result := (others => '0');
        when 1 => result := (others => '1');
        when 2 =>
          for i in result'range loop
            result(i) := '1' when (i mod 2) = 0 else '0';
          end loop;
        when 3 =>
          for i in result'range loop
            result(i) := '1' when ((i + seed) mod 3) = 0 else '0';
          end loop;
        when others =>
          for i in result'range loop
            result(i) := '1' when i = (seed mod width) else '0';
          end loop;
      end case;
      return result;
    end function;

  begin
    for dw_idx in data_widths'range loop
      for uw in 0 to 16 loop
        for seed in 0 to 4 loop
          checks := checks + 1;

          data_field := (others => '0');
          data_field(data_widths(dw_idx) - 1 downto 0) := pattern(seed, data_widths(dw_idx));
          user_field := (others => '0');
          if uw > 0 then
            user_field(uw - 1 downto 0) := pattern(seed + 1, uw);
          end if;
          last_bit := '1' when (seed mod 2) = 0 else '0';

          -- ---- to_slv equivalence ----
          o_m2s := orig_pkg.axi_stream_pkg.axi_stream_m2s_init;
          o_m2s.data(data_widths(dw_idx) - 1 downto 0) := data_field(data_widths(dw_idx) - 1 downto 0);
          o_m2s.last := last_bit;
          if uw > 0 then
            o_m2s.user(uw - 1 downto 0) := user_field(uw - 1 downto 0);
          end if;

          p_m2s := patched_pkg.axi_stream_pkg.axi_stream_m2s_init;
          p_m2s.data(data_widths(dw_idx) - 1 downto 0) := data_field(data_widths(dw_idx) - 1 downto 0);
          p_m2s.last := last_bit;
          if uw > 0 then
            p_m2s.user(uw - 1 downto 0) := user_field(uw - 1 downto 0);
          end if;

          o_slv := (others => '0');
          p_slv := (others => '0');
          o_slv(orig_pkg.axi_stream_pkg.axi_stream_m2s_sz(data_widths(dw_idx), uw) - 1 downto 0) :=
            orig_pkg.axi_stream_pkg.to_slv(o_m2s, data_widths(dw_idx), uw);
          p_slv(patched_pkg.axi_stream_pkg.axi_stream_m2s_sz(data_widths(dw_idx), uw) - 1 downto 0) :=
            patched_pkg.axi_stream_pkg.to_slv(p_m2s, data_widths(dw_idx), uw);

          if o_slv /= p_slv then
            mismatches := mismatches + 1;
            report "to_slv MISMATCH data_width=" & to_string(data_widths(dw_idx)) &
              " user_width=" & to_string(uw) & " seed=" & to_string(seed) &
              " orig=" & to_hstring(o_slv) & " patched=" & to_hstring(p_slv)
              severity error;
          end if;

          -- ---- to_axi_stream_m2s equivalence ----
          o_back := orig_pkg.axi_stream_pkg.to_axi_stream_m2s(
            o_slv(orig_pkg.axi_stream_pkg.axi_stream_m2s_sz(data_widths(dw_idx), uw) - 1 downto 0),
            data_widths(dw_idx), uw, '1'
          );
          p_back2 := patched_pkg.axi_stream_pkg.to_axi_stream_m2s(
            p_slv(patched_pkg.axi_stream_pkg.axi_stream_m2s_sz(data_widths(dw_idx), uw) - 1 downto 0),
            data_widths(dw_idx), uw, '1'
          );

          if o_back.data(data_widths(dw_idx) - 1 downto 0) /= p_back2.data(data_widths(dw_idx) - 1 downto 0) or
             o_back.last /= p_back2.last or
             (uw > 0 and o_back.user(uw - 1 downto 0) /= p_back2.user(uw - 1 downto 0)) then
            mismatches := mismatches + 1;
            report "to_axi_stream_m2s MISMATCH data_width=" & to_string(data_widths(dw_idx)) &
              " user_width=" & to_string(uw) & " seed=" & to_string(seed)
              severity error;
          end if;
        end loop;
      end loop;
    end loop;

    report "tb_axi_stream_pkg_equiv: " & to_string(checks) & " combos checked, " &
      to_string(mismatches) & " mismatches";

    assert mismatches = 0
      report "tb_axi_stream_pkg_equiv: EQUIVALENCE FAILED"
      severity failure;

    report "tb_axi_stream_pkg_equiv: PASS -- patched axi_stream_pkg is bit-exact equivalent to original";

    wait;
  end process;

end architecture;
