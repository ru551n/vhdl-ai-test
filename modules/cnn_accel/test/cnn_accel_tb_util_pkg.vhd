library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library register_file;
use register_file.register_file_pkg.register_t;

library cnn_accel;
use cnn_accel.cnn_accel_register_record_pkg.all;

-- Shared testbench-only utilities for cnn_accel's VUnit testbenches --
-- number-to-string diagnostic rendering, a stall-aware AXI response
-- latency helper, and a common STATUS-register describer -- factored
-- out so no VHDL testbench needs its own copy.
package cnn_accel_tb_util_pkg is

  -- Decimal rendering of an unsigned of any width. Deliberately not
  -- 'to_string(to_integer(...))': a counter register is a full 32-bit
  -- unsigned and values at or above 2**31 do not fit VHDL's signed
  -- 'integer'. Long division by 10 has no such limit.
  function to_dec(value : u_unsigned) return string;
  function to_dec(value : std_ulogic) return string;

  -- Lower case, zero padded, exactly 'num_digits' hex digits.
  function to_hex(value : u_unsigned; num_digits : positive) return string;

  -- A response latency only makes sense together with stalling: with
  -- 'stall_probability_percent' = 0 an AXI slave BFM must be as fast as
  -- it can be, so the cheap no-stall configs stay cheap.
  function axi_response_latency(
    stall_probability_percent : natural;
    clk_period : time
  ) return time;

  -- Renders a STATUS register both raw and field-by-field, for
  -- diagnostic/failure messages. Covers the fields every cnn_accel
  -- STATUS register has (busy/done/error/err_code/err_pc_low); a
  -- testbench with extra fields (e.g. QUEUED) appends them at the call
  -- site.
  function describe_status(
    status_slv : register_t;
    status : cnn_accel_status_t
  ) return string;

end package;

package body cnn_accel_tb_util_pkg is

  function to_dec(value : u_unsigned) return string is
    variable rest : u_unsigned(value'length - 1 downto 0) := value;
    -- 'value'length' decimal digits is always enough: 2**n - 1 has at
    -- most ceil(n * log10(2)) + 1 <= n digits for every n >= 1.
    variable digits : string(1 to value'length) := (others => '0');
    variable idx : natural := value'length;
  begin
    if rest = 0 then
      return "0";
    end if;
    while rest /= 0 loop
      digits(idx) := character'val(character'pos('0') + to_integer(rest mod 10));
      idx := idx - 1;
      rest := rest / 10;
    end loop;
    return digits(idx + 1 to digits'high);
  end function;

  function to_dec(value : std_ulogic) return string is
  begin
    if value = '1' then
      return "1";
    end if;
    return "0";
  end function;

  function to_hex(value : u_unsigned; num_digits : positive) return string is
    constant c_nibbles : string(1 to 16) := "0123456789abcdef";
    constant padded : u_unsigned(4 * num_digits - 1 downto 0) := resize(value, 4 * num_digits);
    variable result : string(1 to num_digits);
  begin
    for i in 0 to num_digits - 1 loop
      result(num_digits - i) := c_nibbles(to_integer(padded(4 * i + 3 downto 4 * i)) + 1);
    end loop;
    return result;
  end function;

  function axi_response_latency(
    stall_probability_percent : natural;
    clk_period : time
  ) return time is
  begin
    if stall_probability_percent = 0 then
      return 0 ns;
    end if;
    return 3 * clk_period;
  end function;

  function describe_status(
    status_slv : register_t;
    status : cnn_accel_status_t
  ) return string is
  begin
    return "STATUS=0x" & to_hex(u_unsigned(status_slv), 8)
      & " (busy=" & to_string(status.busy)
      & ", done=" & to_string(status.done)
      & ", error=" & to_string(status.error)
      & ", err_code=" & to_dec(status.err_code)
      & ", err_pc_low=" & to_dec(status.err_pc_low) & ")";
  end function;

end package body;
